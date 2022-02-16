from torch import nn

def init_weights(m):
	for p in m.parameters():
		p = nn.init.uniform_(p, a=-0.1, b=0.1)
	