import  matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import torch.nn as nn

device = 'cuda'
s_curve , _  = make_s_curve(10**4 , noise = 0.1)
s_curve = s_curve[:,[0,2] ]/10.0

print("shape of moons :",np.shape(s_curve))
dataset = torch.Tensor(s_curve).float().to(device)

# TODO 确定超参数的值
num_steps = 100 # 可以由beta alpha 分布 均值 标准差 进行估算

# 学习的超参数 动态的在（0，1）之间逐渐增大
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)* (0.5e-2 - 1e-5) + 1e-5
betas = betas.to(device)

# 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas ,dim=0 ).to(device)
alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]],0).to(device)
alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
one_minus_alphas_bar_log = torch.log(1-alphas_prod).to(device)
one_minus_alphas_bar_sqrt = torch.sqrt(1-alphas_prod).to(device)

assert  alphas_prod.shape == alphas_prod.shape == alphas_prod_p.shape \
        == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
        == one_minus_alphas_bar_sqrt.shape
print("all the same shape:",betas.shape)

def q_x(x_0 ,t):
    noise = torch.randn_like(x_0).to(device)
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_l_m_t * noise)

class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()
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
    def forward(self,x,t):
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx +1](x)

        x = self.linears[-1](x)
        return x

def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    batch_size = x_0.shape[0]
    t = torch.randint(0,n_steps,size=(batch_size//2,))
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1).to(device)
    a = alphas_bar_sqrt[t]
    e = torch.randn_like(x_0).to(device)
    aml = one_minus_alphas_bar_sqrt[t]
    x = x_0* a + e *aml
    output = model(x.to(device), t.squeeze(-1))
    return (e-output).square().mean()

def p_sample_loop(model ,shape ,n_steps,betas ,one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape).to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x, i ,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor(t).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t].sqrt()) * (x-(coeff * eps_theta)))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)

print('Training model ……')
batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle = True)
num_epoch = 5000
plt.rc('text',color='blue')

model = MLPDiffusion(num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

for t in range(num_epoch):
    for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(
            model,
            batch_x.to(device),
            alphas_bar_sqrt.to(device),
            one_minus_alphas_bar_sqrt.to(device),
            num_steps
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1.)
        optimizer.step()

    if (t% 10 == 0):
        print(loss.item())
        with torch.no_grad():
            x_seq = p_sample_loop(model,dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)

        fig ,axs = plt.subplots(1,10,figsize=(28,3))
        for i in range(1,11):
            cur_x = x_seq[i*10].detach().cpu().numpy()
            axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white');
            axs[i-1].set_axis_off()
            axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        plt.savefig(f'{t}.png')
        plt.close()