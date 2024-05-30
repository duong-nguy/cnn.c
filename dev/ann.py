import torch

X = torch.tensor([[0.911647, 0.197551],
                  [0.335223, 0.768230]], requires_grad=False)

Y = torch.tensor([1.0, 1.0], requires_grad=False)

W = torch.tensor([0.680375, -0.211234], requires_grad=True)
B = torch.tensor([0.0], requires_grad=True)

Y_hat = torch.mv(X, W) + B  # Matrix-vector multiplication
Y_hat = torch.nn.functional.sigmoid(Y_hat)
loss = torch.nn.functional.mse_loss(Y_hat, Y)
loss.backward()

# print(X, Y)
# print(W, B)
print("_________")
print(Y_hat)
print(loss)
print("_________")
# print(W.grad, B.grad, loss.grad)
