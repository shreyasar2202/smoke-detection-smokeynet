import torch

def predict_tile(output):
    tile_preds = torch.sigmoid(output)
    tile_preds = (preds > 0.5).int()
    
    return tile_preds

def predict_image_from_tile_preds(tile_preds):
    """
    Description: Takes predictions per tile and returns if any are true
    Args:
        - tile_preds: 0/1 predictions per tile. Use predict_tile function.
    """
    
    return (torch.sum() > 0).int()