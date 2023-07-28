import sys



def error_message_detail(error, error_details:sys):
    """ Return error message detail
    Args:
        error (str): Error message
        error_details (sys): Error details
    Returns: 
        str: Error message detail
    """
    _, _, exec_tb = error_details.exc_info()
    filename = exec_tb.tb_frame.f_code.co_filename
    line_number = exec_tb.tb_lineno
    error_message = f"{error} in {filename} at line {line_number}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details:sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self) -> str:
        return self.error_message