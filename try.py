    raise type(e)(f'{obj_cls.__name__}: {e}')
KeyError: 'DeformableDETR: "DeformableDETRHead: \'cls_cost\'"'



  File "/home/talimu/anaconda3/envs/pytorch/lib/python3.7/site-packages/torch/_tensor.py", line 307, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/home/talimu/anaconda3/envs/pytorch/lib/python3.7/site-packages/torch/autograd/__init__.py", line 156, in backward
    allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED