import cv2
from dgst.utils.loader import Image
from dgst.utils.validation import ImageValidator, ValidationError

class ImageFuser:

    def __init__(self):
        pass

    def subtract(self, img1: Image, img2: Image) -> Image:
        """Subtract two images pixel-wise."""
        # Precondition validation
        ImageValidator.validate_data_not_none(img1, "ImageFuser.subtract")
        ImageValidator.validate_data_not_none(img2, "ImageFuser.subtract")
        
        if img1.data.shape != img2.data.shape:
            raise ValidationError(
                f"ImageFuser.subtract: Images must have the same shape. "
                f"Got {img1.data.shape} and {img2.data.shape}"
            )
        
        if img1.data.dtype != img2.data.dtype:
            raise ValidationError(
                f"ImageFuser.subtract: Images must have the same dtype. "
                f"Got {img1.data.dtype} and {img2.data.dtype}"
            )
        
        res = img1.clone()
        result_data = cv2.subtract(res.data, img2.data)
        
        # Post-processing validation
        if result_data is None:
            raise ValidationError("ImageFuser.subtract: Result is None")
        if result_data.shape != img1.data.shape:
            raise ValidationError(
                f"ImageFuser.subtract: Result shape {result_data.shape} "
                f"doesn't match input shape {img1.data.shape}"
            )
        
        res.data = result_data
        
        # Update metadata
        res.metadata.add_step({
            "technique": "subtract",
            "operation": "image_fusion",
            "output_shape": res.data.shape
        })
        
        return res
    
    def bitwise_and(self, img1: Image, img2: Image) -> Image:
        """Perform bitwise AND operation between two images."""
        # Precondition validation
        ImageValidator.validate_data_not_none(img1, "ImageFuser.bitwise_and")
        ImageValidator.validate_data_not_none(img2, "ImageFuser.bitwise_and")
        
        if img1.data.shape != img2.data.shape:
            raise ValidationError(
                f"ImageFuser.bitwise_and: Images must have the same shape. "
                f"Got {img1.data.shape} and {img2.data.shape}"
            )
        
        if img1.data.dtype != img2.data.dtype:
            raise ValidationError(
                f"ImageFuser.bitwise_and: Images must have the same dtype. "
                f"Got {img1.data.dtype} and {img2.data.dtype}"
            )
        
        res = img1.clone()
        result_data = cv2.bitwise_and(res.data, img2.data)
        
        # Post-processing validation
        if result_data is None:
            raise ValidationError("ImageFuser.bitwise_and: Result is None")
        if result_data.shape != img1.data.shape:
            raise ValidationError(
                f"ImageFuser.bitwise_and: Result shape {result_data.shape} "
                f"doesn't match input shape {img1.data.shape}"
            )
        
        res.data = result_data
        
        # Update metadata
        res.metadata.add_step({
            "technique": "bitwise_and",
            "operation": "image_fusion",
            "output_shape": res.data.shape
        })
        
        return res
    
    def bitwise_or(self, img1: Image, img2: Image) -> Image:
        """Perform bitwise OR operation between two images."""
        # Precondition validation
        ImageValidator.validate_data_not_none(img1, "ImageFuser.bitwise_or")
        ImageValidator.validate_data_not_none(img2, "ImageFuser.bitwise_or")
        
        if img1.data.shape != img2.data.shape:
            raise ValidationError(
                f"ImageFuser.bitwise_or: Images must have the same shape. "
                f"Got {img1.data.shape} and {img2.data.shape}"
            )
        
        if img1.data.dtype != img2.data.dtype:
            raise ValidationError(
                f"ImageFuser.bitwise_or: Images must have the same dtype. "
                f"Got {img1.data.dtype} and {img2.data.dtype}"
            )
        
        res = img1.clone()
        result_data = cv2.bitwise_or(res.data, img2.data)
        
        # Post-processing validation
        if result_data is None:
            raise ValidationError("ImageFuser.bitwise_or: Result is None")
        if result_data.shape != img1.data.shape:
            raise ValidationError(
                f"ImageFuser.bitwise_or: Result shape {result_data.shape} "
                f"doesn't match input shape {img1.data.shape}"
            )
        
        res.data = result_data
        
        # Update metadata
        res.metadata.add_step({
            "technique": "bitwise_or",
            "operation": "image_fusion",
            "output_shape": res.data.shape
        })
        
        return res