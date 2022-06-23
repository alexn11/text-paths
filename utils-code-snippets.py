  # https://stackoverflow.com/questions/52988876/how-can-i-visualize-what-happens-during-loss-backward
  def getBack(self, var_grad_fn):
    #print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                #print(n[0])
                #print('Tensor with grad found:', tensor)
                #print(' - gradient:', tensor.grad)
                #print()
                #print(f'device: {tensor.device}')
                assert(tensor.device == torch.device('cuda:0'))
            except AttributeError as e:
                self.getBack(n[0])
