import torch

import torch

X = torch.tensor([[0.513401, 0.952230, 0.916195, 0.635712],
                  [0.717297, 0.141603, 0.606969, 0.016301]], requires_grad=False)

Y = torch.tensor([1.0, 1.0], requires_grad=False)

W = torch.tensor([-0.444451, 0.107940, -0.045206, 0.257742], requires_grad=True)
B = torch.tensor([-0.270431], requires_grad=True)

Y_hat = torch.mv(X, W) + B  # Matrix-vector multiplication
loss = torch.nn.functional.mse_loss(Y_hat, Y)
loss.backward()

print(X, Y)
print(W, B)
print(Y_hat, loss)
print(W.grad, B.grad, loss.grad)
