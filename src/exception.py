import sys  #used to handle exceptions i.e check on runtime environment
from src.logger import logging  #logger is imported so that exceptionn can be recorded in log file

def error_detail_message(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    error_msg="Error occured"
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error meessage [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message
    
class customException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_detail_message(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message