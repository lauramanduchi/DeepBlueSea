import numpy as np

def boxes_per_image(class_score, reg_score, n_proposals=2000, nms_thres=0.2, debug=0):
    '''
    outputs the coordinates of the bounding boxes as [x1, x2, y1, y2]

    :param class_score: input class scores per image as per output by the model [768, 768, n_anchors, 2]
    :param reg_score: input reg scores per image as per output by the model [768, 768, n_anchors, 2]
    :return: list of coordinates and list of scores where coordinates are [x1, x2, y1, y2]
    '''
    # Get n_prosal'th biggest probability
    prob_class = class_softmax(class_score)[:, :, :, 0]
    # Sort propabilities in asceding order [0.0, 0.1, ..., 1.0]
    sorted_probs = prob_class.flatten().sort()
    if len(sorted_probs) < n_proposals:
        class_thresh = min(sorted_probs)
    else:
        class_thresh = sorted_probs[-n_proposals]

    neg_mask = prob_class < class_thresh
    if debug:
        print("num of pos anchors:", 768 ** 2 * 10 - np.sum(neg_mask))
    # only look at positive labels (or labels above threshold)
    prob_class[neg_mask] = 0
    c_y, c_x, anchor_dim = np.nonzero(prob_class)
    w = reg_score[c_y, c_x, anchor_dim][:, 0]
    h = reg_score[c_y, c_x, anchor_dim][:, 1]
    p = prob_class[c_y, c_x, anchor_dim]
    trial_boxes = np.zeros((len(c_x), 4))
    trial_boxes[:, 0] = c_x - w / 2
    trial_boxes[:, 1] = c_x + w / 2
    trial_boxes[:, 2] = c_y - h / 2
    trial_boxes[:, 3] = c_y + h / 2
    boxes, to_keep_indexs = non_max_suppression_fast(trial_boxes, nms_thres)
    p = p[to_keep_indexs]

    return {'boxes': list(boxes),
            'scores': list(p)}


def class_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    shapes = np.shape(x)
    if len(shapes) != 4:
        print("error! was expecting array of shape 4 [768, 768, n_anchors, 2]")
    nom = np.exp(x)
    denom = np.reshape(np.sum(nom, axis=-1), (shapes[0], shapes[1], shapes[2], 1))
    return nom / np.tile(denom, (1, 1, 1, 2))


# internet code for NMS
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# import the necessary packages

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), pick