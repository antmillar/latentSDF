import torch
import torch.nn as nn
import torch.nn.functional as F


class deepSDFCodedShape(nn.Module):

    def __init__(self, input_dim = 2, code_dim = 2, hidden_dim = 256, num_layers = 8, skip_layers = [4]):

      super().__init__()  

      self.skip_layers = skip_layers
      self.num_layers = num_layers
      self.linears = nn.ModuleList()
      self.linears.append(nn.Linear(input_dim + code_dim, hidden_dim))

      for i in range(1, num_layers):
        #skip layers
        if( i + 1 in skip_layers):
          self.linears.append(nn.Linear(hidden_dim, hidden_dim - (input_dim + code_dim)))
        
        #final layer
        elif(i == num_layers - 1):
          self.linears.append(nn.Linear(hidden_dim, 1))
        
        #standard linear layers
        else:
          self.linears.append(nn.Linear(hidden_dim, hidden_dim))

      self.tanh = nn.Tanh()

    def forward(self, shape_code, coord):

      shape_code = shape_code.float()
      coord = coord.float()

      shape_code = shape_code.repeat(coord.shape[0], 1)
      input = torch.cat((shape_code, coord), dim = 1)
      x = input

      for i, layer in enumerate(self.linears):
        x = layer(x)

        if(i == self.num_layers - 1):
          x = self.tanh(x)
        else:
          x = F.relu(x)

        if(i + 1 in self.skip_layers):  
          x = torch.cat((x, input), dim = 1) #skip connection

      return x