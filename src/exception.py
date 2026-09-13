class CreditPipelineError(Exception):
    """Base exception class for the credit scoring pipeline"""
    pass


class DataValidationError(CreditPipelineError):
    """Raised when input data failed validation checks"""
    pass
