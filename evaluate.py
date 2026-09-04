"""
    evaluate.py
    IIT : Istituto italiano di tecnologia
    Center for Cultural Heritage Technology (CCHT) research line
    
    Description: This repo contains the code for our paper 
    Deep Learning for Archaeological Object Detection on LiDAR: New Evaluation Measures and Insights, 
    Published in MDPI Remote Sensing on 31 March 2022

    Disclaimer:
    The information and content provided by this application is for information purposes only.
    The software is provided "as is", without warranty of any kind, express or implied,
    including but not limited to the warranties of merchantability,
    fitness for a particular purpose and noninfringement. In no event shall the authors,
    CCHT or IIT be liable for any claim, damages or other liability, whether in an action of contract,
    tort or otherwise, arising from, out of or in connection with the software
    or the use or other dealings in the software.
    
    LICENSE:
    The information and content provided by this application is for information purposes only. 
    The software is provided "as is", without warranty of any kind, express or implied, 
    including but not limited to the warranties of merchantability, 
    fitness for a particular purpose and noninfringement. In no event shall the authors, 
    CCHT or IIT be liable for any claim, damages or other liability, whether in an action of contract, 
    tort or otherwise, arising from, out of or in connection with the software 
    or the use or other dealings in the software.
"""

import copy

import numpy as np
import shapely.wkt


def compute_iou_check_threshold(pred_polygon, gt_polygon, iou_threshold):
    """Computes IoU through compute_iou and checks it againts the given threshold

        returns false if comparing with np.nan 
        (i.e. both intersection and union are zero, or error occurred)

    Parameters
    ----------
    pred_polygon : shapely.geometry.polygon.Polygon
        a predicted polygon shape
    gt_polygon : shapely.geometry.polygon.Polygon
        a ground-truth polygon shape
    iou_threshold : float
        threshold for IoU result comparison 

    Returns
    -------
    bool
        true if computed iou above threshold, false otherwise (see description)
    """
    return compute_iou(pred_polygon, gt_polygon) >= iou_threshold


def compute_iou(pred_polygon, gt_polygon):
    """Computes Intersection over Union (IoU) leveraging shapely functions
    Makes use of a small arbitrary constant to avoid division by zero

    returns nan if both intersection and union are zero, or if error occurs

    Parameters
    ----------
    pred_polygon : shapely.geometry.polygon.Polygon
         a predicted polygon shape
    gt_polygon : shapely.geometry.polygon.Polygon
        a ground-truth polygon shape

    Returns
    -------
    float
        the computed IoU amount or nan if both intersection and union are zero, or if error occurs
    """
    SMOOTH = 1e-6
    iou = np.nan
    try:
        intersection = pred_polygon.intersection(gt_polygon).area
        union = pred_polygon.union(gt_polygon).area
        if(union != 0 or intersection != 0):
            iou = ((intersection +  SMOOTH) / (union +  SMOOTH))
    except Exception:
        iou = np.nan
    return iou

def compute_iou_metric_centroid(metadata_thing_classes, pred_shapes, gt_shapes, iou_threshold):
    """
    Compute the centroid based mesure considering the spatial relationship between predicted and ground truth objects.

    Checks if the centroid of each predicted object’s bounding box falls inside the closest ground truth bounding box.
    Conversely, if the predicted box's centroid does not fall into any ground truth polygon (or its IoU is below threshold), 
    is considered as a false positive.
    Finally, it computes false negatives, getting it from ground truth objects unbounded to true positives.

    Parameters
    ----------
    metadata_thing_classes : List[str]
        list of class names of archaeological objects 
    pred_shapes : Dict[str, shapely.geometry.multipolygon.MultiPolygon]
        per-class dictionary, mapping each class name to the collection of shapely polygon relative to the 
        predicted elements
    gt_shapes : Dict[str, shapely.geometry.multipolygon.MultiPolygon]
        per-class dictionary, mapping each class name to the collection of shapely polygon relative to the 
        ground truth elements 
    iou_threshold : float
        threshold for IoU result comparison: value above this thresholds are candidates to true positive assignment

    Returns
    -------
    Dict[str, Dict[str, int]]
        per-class nested dictionary, containing for each class name integer computed values of 
        true positives (TP)
        false positives (FP)
        false negatives (FN)
    """
    res = {}
    for cl in metadata_thing_classes:
        res[cl] = {}
        res[cl]["TP"] = 0
        res[cl]["FP"] = 0
        res[cl]["FN"] = 0
        
        # deepcopy to allow alteration on predicted obj list
        pred_shapes_cl = copy.deepcopy(list(pred_shapes[cl]))
        gt_shapes_cl = copy.deepcopy(list(gt_shapes[cl]))

        # map object to string repr, in order to make them hashable
        pred_shapes_cl = [p.wkt for p in pred_shapes_cl]
        gt_shapes_cl = [g.wkt for g in gt_shapes_cl]

        # loop on predicted bbox

        # check if predicted box's centroid falls into any GT polygon and iou threshold is above threshold
        # if yes, and it has not been previously counted:
        #       increment TP
        # if predicted box's centroid does not fall into any GT polygon (or its iou is below threshold)
        #       increment FP
        tp_shapes = set()
        fp_shapes = set()
        for p in pred_shapes_cl:
            for s in gt_shapes_cl:
                if(shapely.wkt.loads(s).contains(shapely.wkt.loads(p).centroid) 
                and compute_iou_check_threshold(shapely.wkt.loads(p), shapely.wkt.loads(s), 
                iou_threshold)):
                    tp_shapes.add(p)
                    gt_shapes_cl.remove(s) # avoid considering this polygon for next iterations
                    
        for p in pred_shapes_cl:
            if(p not in tp_shapes):
                fp_shapes.add(p)
                
        res[cl]["TP"] = res[cl]["TP"] + len(tp_shapes)
        res[cl]["FP"] = res[cl]["FP"] + len(fp_shapes)

    # conversely, consider the GT annotations to check if there are FN
    # loop on GT annotations

    for cl in metadata_thing_classes:
        
        # deepcopy to allow alteration on predicted obj list
        pred_shapes_cl = copy.deepcopy(list(pred_shapes[cl]))
        gt_shapes_cl = copy.deepcopy(list(gt_shapes[cl]))

        # map object to string repr, in order to make them hashable
        pred_shapes_cl = [p.wkt for p in pred_shapes_cl]
        gt_shapes_cl = [g.wkt for g in gt_shapes_cl]
        
        res[cl]["FN"] = 0
        tp_shapes = set()
        fn_shapes = set()
        for s in gt_shapes_cl:
            for p in pred_shapes_cl:
                if(shapely.wkt.loads(s).contains(shapely.wkt.loads(p).centroid) 
                and compute_iou_check_threshold(shapely.wkt.loads(p), shapely.wkt.loads(s), 
                iou_threshold)):
                    tp_shapes.add(s)
                    pred_shapes_cl.remove(p) # avoid considering this polygon for next iterations
        for s in gt_shapes_cl:
            if(s not in tp_shapes):
                fn_shapes.add(s)
        res[cl]["FN"] = res[cl]["FN"] + len(fn_shapes)
    return res


def compute_metrics(class_mask, class_pred):
    """ Compute the pixel-wise measure, considering the object detection task at a pixel-level, thus comparing ground truth and predicted masks pixel by pixel.

    Parameters
    ----------
    class_mask : numpy.ndarray
        binary (boolean) image mask of ground truth objects. All the items (relative to a single class) must insist on the same mask. 
    class_pred : numpy.ndarray
        binary (boolean) image mask of predicted objects. All the prediction (relative to a single class) must insist on the same mask. 

    Returns
    -------
    Tuple[numpy.int64, numpy.int64, numpy.int64, numpy.int64]
        total amount of TP, FP, FN and TN pixels respectively
    """
    tp_tensor = np.bitwise_and(class_mask, class_pred)
    fp_tensor = np.bitwise_and(np.invert(class_mask), class_pred)
    fn_tensor = np.bitwise_and(class_mask, np.invert(class_pred))
    tn_tensor = np.bitwise_and(np.invert(class_mask), np.invert(class_pred))
    return tp_tensor.sum(), fp_tensor.sum(), fn_tensor.sum(), tn_tensor.sum()


def iou_simple(pred_t, mask_t, smooth=1e-6):
    """ Computes Intersection over Union (IoU) leveraging numpy functions
    Makes use of a small arbitrary constant to avoid division by zero

    returns nan if both intersection and union are zero, or if error occurs

    Parameters
    ----------
    pred_t : numpy.ndarray
        binary (boolean) image mask of predicted objects. All the prediction (relative to a single class) must insist on the same mask.
    mask_t : numpy.ndarray
        binary (boolean) image mask of ground truth objects. All the items (relative to a single class) must insist on the same mask.
    smooth : float, optional
        small arbitrary constant to avoid division by zero, by default 1e-6

    Returns
    -------
    float
        the computed IoU amount or nan if both intersection and union are zero, or if error occurs
    """
    SMOOTH = smooth
    iou = np.nan
    try:
        intersection =  np.bitwise_and(pred_t, mask_t).sum()
        union = np.bitwise_or(pred_t, mask_t).sum()
        if(union == 0 and intersection == 0):
            iou = np.nan
        else:
            iou = ((intersection +  SMOOTH) / (union +  SMOOTH)).item()
    except Exception:
        iou = np.nan
    return iou
