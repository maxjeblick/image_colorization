from keras.losses import binary_crossentropy
import keras.backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


def focus_loss(y_true, y_pred):
    gamma = 0.5  # 2.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    loss = y_true * K.log(y_pred + K.epsilon()) * (1 - y_pred + K.epsilon()) ** gamma + \
           (1 - y_true) * K.log(1 - y_pred + K.epsilon()) * (y_pred + K.epsilon()) ** gamma
    return -K.mean(loss)


def focal_dice_loss(y_true, y_pred):
    return focus_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)


def iou_loss_core(true, pred):
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return K.sum(intersection, axis=-1) / (K.sum(union, axis=-1) + K.epsilon())


def mixedPenalty(iouWeight=0.5, penaltyWeight=2):
    def loss(y_true, y_pred):
        #flattening each image - resulting shape = (batch, totalPixels)
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)

        #boolean filters for ground trugh images with and without masks
        noMask = K.sum(y_true, axis=-1, keepdims=True)  # has shape (batch_size, 1)
        noMask = castF(K.less(noMask, .9))
        hasMask = 1 - noMask

        #regular binary crossentropy for each image
        cross = binary_crossentropy(y_true, y_pred)

        #for ground truth images with no mask, apply a penalty
        noMaskLoss = penaltyWeight * (noMask * cross)
        #get the iou loss (negative because we want it to decrease)
        iouLoss = -iou_loss_core(y_true, y_pred)

        #sum all losses, filtered by the presence of ground true masks
        return noMaskLoss + hasMask * (cross + (iouWeight * iouLoss))

    return loss


def penality_mixed_bce_dice(y_true, y_pred):

    return mixedPenalty()(y_true, y_pred) + bce_dice_loss(y_true, y_pred)
