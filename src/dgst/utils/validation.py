"""
Validation utilities for image processing pipeline.

This module provides comprehensive validation functions to ensure data integrity
throughout the processing pipeline, including format validation, dimension checks,
and type verification.
"""

import numpy as np
from typing import Tuple, Optional, List
from enum import Enum

from dgst.utils.loader import Image, ImageFormat


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class ImageValidator:
    """Validator class for comprehensive image validation."""

    @staticmethod
    def validate_format(image: Image, expected_format: ImageFormat, step_name: str) -> None:
        """
        Validate that the image has the expected format.

        Args:
            image: Image object to validate
            expected_format: Expected ImageFormat
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If format doesn't match expected format
        """
        if image.format != expected_format:
            raise ValidationError(
                f"{step_name}: Expected format {expected_format.name}, "
                f"but got {image.format.name}"
            )

    @staticmethod
    def validate_multiple_formats(
        image: Image, 
        expected_formats: List[ImageFormat], 
        step_name: str
    ) -> None:
        """
        Validate that the image has one of the expected formats.

        Args:
            image: Image object to validate
            expected_formats: List of acceptable ImageFormat values
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If format doesn't match any expected format
        """
        if image.format not in expected_formats:
            format_names = [fmt.name for fmt in expected_formats]
            raise ValidationError(
                f"{step_name}: Expected one of formats {format_names}, "
                f"but got {image.format.name}"
            )

    @staticmethod
    def validate_data_not_none(image: Image, step_name: str) -> None:
        """
        Validate that image data is not None.

        Args:
            image: Image object to validate
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If image data is None
        """
        if image.data is None:
            raise ValidationError(f"{step_name}: Image.data is None")

    @staticmethod
    def validate_data_type(
        image: Image, 
        expected_dtype: np.dtype, 
        step_name: str
    ) -> None:
        """
        Validate that image data has the expected numpy dtype.

        Args:
            image: Image object to validate
            expected_dtype: Expected numpy dtype
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If dtype doesn't match
        """
        ImageValidator.validate_data_not_none(image, step_name)
        if image.data.dtype != expected_dtype:
            raise ValidationError(
                f"{step_name}: Expected dtype {expected_dtype}, "
                f"but got {image.data.dtype}"
            )

    @staticmethod
    def validate_data_dimensions(
        image: Image, 
        expected_ndim: int, 
        step_name: str
    ) -> None:
        """
        Validate that image data has the expected number of dimensions.

        Args:
            image: Image object to validate
            expected_ndim: Expected number of dimensions (2 or 3)
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If dimensions don't match
        """
        ImageValidator.validate_data_not_none(image, step_name)
        if image.data.ndim != expected_ndim:
            raise ValidationError(
                f"{step_name}: Expected {expected_ndim}D array, "
                f"but got {image.data.ndim}D array with shape {image.data.shape}"
            )

    @staticmethod
    def validate_grayscale_image(image: Image, step_name: str) -> None:
        """
        Validate that image is in grayscale format with correct dimensions and type.

        Args:
            image: Image object to validate
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If image is not a valid grayscale image
        """
        ImageValidator.validate_format(image, ImageFormat.GRAYSCALE, step_name)
        ImageValidator.validate_data_not_none(image, step_name)
        ImageValidator.validate_data_dimensions(image, 2, step_name)
        ImageValidator.validate_data_type(image, np.uint8, step_name)

    @staticmethod
    def validate_bgr_image(image: Image, step_name: str) -> None:
        """
        Validate that image is in BGR format with correct dimensions and type.

        Args:
            image: Image object to validate
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If image is not a valid BGR image
        """
        ImageValidator.validate_format(image, ImageFormat.BGR, step_name)
        ImageValidator.validate_data_not_none(image, step_name)
        ImageValidator.validate_data_dimensions(image, 3, step_name)
        
        if image.data.shape[2] != 3:
            raise ValidationError(
                f"{step_name}: BGR image must have 3 channels, "
                f"but got {image.data.shape[2]} channels"
            )
        
        ImageValidator.validate_data_type(image, np.uint8, step_name)

    @staticmethod
    def validate_boolean_image(image: Image, step_name: str) -> None:
        """
        Validate that image is in boolean format with correct dimensions.

        Args:
            image: Image object to validate
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If image is not a valid boolean image
        """
        ImageValidator.validate_format(image, ImageFormat.BOOLEAN, step_name)
        ImageValidator.validate_data_not_none(image, step_name)
        ImageValidator.validate_data_dimensions(image, 2, step_name)
        
        if image.data.dtype != np.bool_:
            raise ValidationError(
                f"{step_name}: Boolean image must have dtype bool, "
                f"but got {image.data.dtype}"
            )

    @staticmethod
    def validate_hsv_channels(
        image: Image, 
        step_name: str,
        expected_channels: int = 3
    ) -> None:
        """
        Validate that image has HSV channels with correct properties.

        Args:
            image: Image object to validate
            step_name: Name of the processing step for error messages
            expected_channels: Expected number of HSV channels (default 3)

        Raises:
            ValidationError: If HSV channels are invalid
        """
        if image.hsv_channels is None:
            raise ValidationError(
                f"{step_name}: Image does not contain HSV channels"
            )
        
        if len(image.hsv_channels) < expected_channels:
            raise ValidationError(
                f"{step_name}: Expected at least {expected_channels} HSV channels, "
                f"but got {len(image.hsv_channels)}"
            )
        
        # Validate each channel
        for idx, channel in enumerate(image.hsv_channels[:expected_channels]):
            if not isinstance(channel, np.ndarray):
                raise ValidationError(
                    f"{step_name}: HSV channel {idx} is not a numpy array"
                )
            
            if channel.ndim != 2:
                raise ValidationError(
                    f"{step_name}: HSV channel {idx} must be 2D, "
                    f"but got {channel.ndim}D with shape {channel.shape}"
                )
            
            if channel.dtype != np.uint8:
                raise ValidationError(
                    f"{step_name}: HSV channel {idx} must be uint8, "
                    f"but got {channel.dtype}"
                )

    @staticmethod
    def validate_calibration(image: Image, step_name: str) -> None:
        """
        Validate that image has calibration data.

        Args:
            image: Image object to validate
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If calibration is missing
        """
        if image.calibration is None:
            raise ValidationError(
                f"{step_name}: Image does not contain calibration data"
            )

    @staticmethod
    def validate_calibration_type(
        image: Image, 
        expected_type: str, 
        step_name: str
    ) -> None:
        """
        Validate that calibration has the expected camera type.

        Args:
            image: Image object to validate
            expected_type: Expected camera type (e.g., "kannala")
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If calibration type doesn't match
        """
        ImageValidator.validate_calibration(image, step_name)
        
        if image.calibration.camera_type != expected_type:
            raise ValidationError(
                f"{step_name}: Expected camera type '{expected_type}', "
                f"but got '{image.calibration.camera_type}'"
            )

    @staticmethod
    def validate_image_shape(
        image: Image,
        expected_shape: Optional[Tuple[int, ...]], 
        step_name: str
    ) -> None:
        """
        Validate that image data has the expected shape.

        Args:
            image: Image object to validate
            expected_shape: Expected shape tuple (None values are wildcards)
            step_name: Name of the processing step for error messages

        Raises:
            ValidationError: If shape doesn't match
        """
        ImageValidator.validate_data_not_none(image, step_name)
        
        if expected_shape is not None:
            actual_shape = image.data.shape
            if len(actual_shape) != len(expected_shape):
                raise ValidationError(
                    f"{step_name}: Expected shape with {len(expected_shape)} dimensions, "
                    f"but got {len(actual_shape)} dimensions"
                )
            
            for idx, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if expected is not None and expected != actual:
                    raise ValidationError(
                        f"{step_name}: Expected shape dimension {idx} to be {expected}, "
                        f"but got {actual}"
                    )


class FormatConverter:
    """Utility class for converting between different image formats."""

    @staticmethod
    def to_uint8_if_needed(data: np.ndarray, step_name: str) -> np.ndarray:
        """
        Convert image data to uint8 if it's in float format.

        Args:
            data: Input numpy array
            step_name: Name of the processing step for error messages

        Returns:
            uint8 numpy array

        Raises:
            ValidationError: If conversion is not possible
        """
        if data.dtype == np.uint8:
            return data
        
        if data.dtype in [np.float32, np.float64]:
            # Check if normalized [0, 1] or [0, 255]
            max_val = np.max(data)
            if max_val <= 1.0:
                return (data * 255.0).astype(np.uint8)
            else:
                return np.clip(data, 0, 255).astype(np.uint8)
        
        raise ValidationError(
            f"{step_name}: Cannot convert dtype {data.dtype} to uint8"
        )

    @staticmethod
    def to_float32_if_needed(data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Convert image data to float32.

        Args:
            data: Input numpy array
            normalize: If True and input is uint8, normalize to [0, 1]

        Returns:
            float32 numpy array
        """
        if data.dtype == np.float32:
            return data
        
        if data.dtype == np.uint8:
            if normalize:
                return data.astype(np.float32) / 255.0
            else:
                return data.astype(np.float32)
        
        if data.dtype == np.float64:
            return data.astype(np.float32)
        
        return data.astype(np.float32)

    @staticmethod
    def ensure_contiguous(data: np.ndarray) -> np.ndarray:
        """
        Ensure array is contiguous in memory (required for C functions).

        Args:
            data: Input numpy array

        Returns:
            Contiguous numpy array
        """
        if not data.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(data)
        return data
