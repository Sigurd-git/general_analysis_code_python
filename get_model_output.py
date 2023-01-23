
def get_model_output(model, input_data,cpu=True):
    # model: pytorch model
    # input_data: pytorch tensor
    # return: pytorch tensor

    if cpu:
        model = model.cpu()
    else:
        model = model.cuda()
    model.eval()
    output = model(input_data)
    return output