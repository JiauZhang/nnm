import  matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
import torch.nn as nn

datapoints , _  = make_swiss_roll(1000, noise=0.2)
datapoints = datapoints[:, [0,2]] / 15

plt.scatter(datapoints[:, 0], datapoints[:, 1], s=2, c='r')
plt.savefig('datapoints.png')
plt.close()

dataset = torch.Tensor(datapoints).float()
num_steps = 120
betas = torch.linspace(1e-5, 5e-3, num_steps)
betas = betas
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas ,dim=0 )
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1-alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1-alphas_prod)

def q_x(x_0 ,t):
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_l_m_t * noise

class Model(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(Model,self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(2,num_units),
            nn.ReLU(),
            nn.Linear(num_units,num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),
        ])
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps,num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units)
        ])
    def forward(self, x, t):
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx +1](x)

        x = self.linears[-1](x)
        return x

def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    batch_size = x_0.shape[0]
    t = torch.randint(0, n_steps, size=(batch_size,))
    t = t.unsqueeze(-1)
    a = alphas_bar_sqrt[t]
    e = torch.randn_like(x_0)
    aml = one_minus_alphas_bar_sqrt[t]
    x = x_0* a + e *aml
    output = model(x, t.squeeze(-1))
    return (e-output).square().mean()

def p_sample_loop(model ,shape ,n_steps,betas ,one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x, i ,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor(t)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t]).sqrt() * (x-(coeff * eps_theta)))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

print('Training model...')
batch_size = 512
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 15000
plt.rc('text',color='blue')
model = Model(num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
sum_loss = 0
step = 0

for epoch in range(num_epoch+1):
    for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(
            model,
            batch_x,
            alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt,
            num_steps
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        step += 1
        sum_loss += loss.item()

    if (epoch % 100 == 0):
        print(epoch, sum_loss / step)
        with torch.no_grad():
            x_seq = p_sample_loop(
                model,
                dataset.shape,
                num_steps,
                betas,
                one_minus_alphas_bar_sqrt
            )

        fig, axs = plt.subplots(3, 4, figsize=(10, 10))
        for i in range(12):
            row = i // 4
            col = i % 4
            seq_idx = i * 10 + 9
            cur_x = x_seq[seq_idx].detach().cpu().numpy()
            axs[row][col].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white')
            axs[row][col].set_axis_off()
            axs[row][col].set_title('$q(\mathbf{x}_{'+str(seq_idx)+'})$')
        plt.savefig(f'{epoch}.png')
        plt.close()