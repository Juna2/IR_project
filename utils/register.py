
# Utility function to set parameters of functions.
# Useful to define what consist of a model, dataset, etc.
def wrap_setattr(attr, value):
    def foo(func):
        setattr(func, attr, value)
        return func
    return foo

def setmodelname(value):
    return wrap_setattr('_MODEL_NAME', value)

def setdatasetname(value):
    return wrap_setattr('_DG_NAME', value)


def IoU_arg_setting(func):
    def decorated(*args, **kwargs):
        
        ori_lambda_for_final = args[-1].lambda_for_final
        args[-1].lambda_for_final = 1
        print('Warning!!! : Followings are fixed in get_IoU function \n  lambda_for_final = 1\n  mask_data_size = 224\n  loss_type = "None"\n  R_process = None')
        
        ori_loss_type = args[-1].loss_type
        args[-1].loss_type = 'None'
        
        ori_mask_data_size = args[-1].mask_data_size
        args[-1].mask_data_size = 224
        
        ori_R_process = args[-1].R_process
        args[-1].R_process = None # None
        
        func(*args, **kwargs)
        
        args[-1].lambda_for_final = ori_lambda_for_final
        args[-1].loss_type = ori_loss_type
        args[-1].mask_data_size = ori_mask_data_size
        args[-1].R_process = ori_R_process
        
        return func
    return decorated


































