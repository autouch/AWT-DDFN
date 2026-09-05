import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LearnableMorlet1D(nn.Module):
    """
    Learnable 1D Morlet Wavelet Transform Layer
    Parameters are optimized during training for specific tasks
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_wavelets: int = 32,
                 kernel_size: int = 128,
                 sample_rate: float = 64000,
                 learn_center_freq: bool = True,
                 use_complex: bool = True,):
        """
        Args:
            in_channels: Number of input channels
            num_wavelets: Number of wavelet filters (output channels)
            kernel_size: Size of the wavelet kernel
            sample_rate: Sampling rate of the input signal
            learn_center_freq: Whether to make center frequency learnable
            use_complex: Whether to use complex-valued wavelets
        """
        super(LearnableMorlet1D, self).__init__()
        
        self.in_channels = in_channels
        self.num_wavelets = num_wavelets
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.use_complex = use_complex
        self.learn_center_freq = learn_center_freq
        #self.sigma = nn.Parameter(torch.ones(num_wavelets))
        # Create time vector centered at zero
        t_values = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size).float() / sample_rate
        self.register_buffer('t', t_values)
        min_freq = 1e-6
        max_freq = sample_rate/2.0
        # Initialize center frequencies (logarithmically spaced)
        center_freqs = torch.linspace(min_freq, max_freq, num_wavelets, dtype=torch.float32)
        # Initialize bandwidth parameters based on Q factor

        self.register_buffer('center_freqs', center_freqs)
        # Make parameters learnable based on settings
        if learn_center_freq:
            self.diff_freq = nn.Parameter(torch.diff(center_freqs))
            self.low_freq = nn.Parameter(torch.tensor([min_freq], dtype=torch.float32))
            #self.center_freq=nn.Parameter(center_freqs)

        if use_complex:
            #self.conv=nn.Conv1d(num_wavelets*2,num_wavelets,1,1,0,1,num_wavelets)
            None

    
    def _compute_morlet_wavelet(self, center_freqs: torch.Tensor) -> torch.Tensor:
        """
        Compute Morlet wavelet for given parameters
        
        Args:
            center_freqs: Center frequencies of the wavelets
            sigmas: Standard deviations of the Gaussian envelopes
            
        Returns:
            Complex Morlet wavelets
        """
        # Ensure positive sigmas for numerical stability
        #sigmas = torch.clamp(sigmas, min=1e-6)
        
        # Reshape for broadcasting
        t = self.t.unsqueeze(0)  # (1, kernel_size)
        center_freqs = center_freqs.unsqueeze(1)  # (num_wavelets, 1)

        #sigma = self.sigma.unsqueeze(1)
        # Normalization constant
        normalization = math.pi**(-1/4)  # (num_wavelets, 1)
        
        # Complex sinusoid component
        # Use real-valued operations to avoid complex number issues
        angle =  2*math.pi*center_freqs * t  # (num_wavelets, kernel_size)
        real_wave = torch.cos(angle)  # Real part
        imag_wave = torch.sin(angle)  # Imaginary part
        
        # Gaussian envelope
        #gaussian_env = torch.exp(-t**2 / (2 * sigma ** 2))  # (num_wavelets, kernel_size)
        gaussian_env = torch.exp(-t**2 / (2))  # (num_wavelets, kernel_size)
        # Apply normalization and Gaussian envelope
        real_wavelet = normalization * real_wave * gaussian_env
        imag_wavelet = normalization * imag_wave * gaussian_env
        
        
        return real_wavelet, imag_wavelet
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable Morlet wavelet transform to input signal
        
        Args:
            x: Input tensor of shape (batch, channels, time_steps)
            
        Returns:
            Wavelet coefficients of shape (batch, num_wavelets, time_steps) for magnitude
            or (batch, 2*num_wavelets, time_steps) for complex (real/imag concatenated)
        """
        if self.learn_center_freq:
            diff_freq = torch.cat([self.low_freq, self.diff_freq])
            center_freqs = torch.cumsum(diff_freq, dim=0)
            #center_freqs=self.center_freq
        # Ensure parameters are positive
        else:
            center_freqs = torch.clamp(self.center_freqs, min=1e-6)
        # Generate wavelets for all frequencies at once using vectorized operations
        real_kernels, imag_kernels = self._compute_morlet_wavelet(center_freqs)
        
        
        # Add input channel dimension
        real_kernels = real_kernels.unsqueeze(1)  # (num_wavelets, 1, kernel_size)
        imag_kernels = imag_kernels.unsqueeze(1)  # (num_wavelets, 1, kernel_size)
        self.real_kernels_cache = real_kernels.detach()
        self.imag_kernels_cache = imag_kernels.detach()
        # Apply convolution to input
        # Use 'same' padding to maintain time dimension
        padding = self.kernel_size // 2
        
        if self.use_complex:
            # For complex output, we'll concatenate real and imaginary parts along channel dimension
            # Initialize output tensors
            
            # Convolve with real and imaginary kernels
            conv_real = F.conv1d(x, real_kernels, padding=padding,bias=None)
            conv_imag = F.conv1d(x, imag_kernels, padding=padding,bias=None)
            output=conv_imag+conv_real
        else:
            # For magnitude output
            
            # Process each input channel
            
            # Convolve with real and imaginary kernels
            conv_real = F.conv1d(x, real_kernels, padding=padding)
            conv_imag = F.conv1d(x, imag_kernels, padding=padding)
            
            # Compute magnitude and sum across input channels
            magnitude = torch.sqrt(conv_real**2 + conv_imag**2)
                
            output = magnitude

        return output

# Example usage and test
if __name__ == "__main__":
    # Test parameters
    import pandas as pd
    import matplotlib.pyplot as plt
    '''df = pd.read_csv('data/usedpucsv/N15_M01_F10_pro/K001.csv')  # 替换为你的CSV文件名

    # 获取第一行数据（一维信号）
    first_row = torch.from_numpy(df.iloc[1100].values).float()[0:512]
  # 按行索引获取第一行

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.plot(first_row, linewidth=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('0.png')
    
    # Create the wavelet transform layer
    wavelet_transform = LearnableMorlet1D(1,4,127,sample_rate=64000,min_freq=1e-6,max_freq=32000,use_complex=True)
    
    # Create test input: (B, 1, T)
    real,imag,x = wavelet_transform(first_row.unsqueeze(0).unsqueeze(1))
    i=0
    x=x.squeeze()
    real=real.squeeze()
    imag=imag.squeeze()
    for a in real:
        print(i)
        print(a)
        i=i+1
        plt.figure(figsize=(10, 4))
        plt.plot(a .detach().numpy(), linewidth=0.5)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(str(i)+'.png')
        plt.close()
    print('real')
    for a in imag:
        print(i)
        i=i+1
        plt.figure(figsize=(10, 4))
        plt.plot(a.detach().numpy(), linewidth=0.5)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(str(i)+'.png')
        plt.close()
    print('imag')
    for a in x:
        print(i)
        i=i+1
        plt.figure(figsize=(10, 4))
        plt.plot(a.detach().numpy(), linewidth=0.5)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(str(i)+'.png')
        plt.close()
    print('end')
    # Apply wavelet transform'''
    model = LearnableMorlet1D(1,64,127,sample_rate=64000,use_complex=True)
    x = torch.randn(1, 1, 2048)
    _ = model(x) 
    real=model.real_kernels_cache.squeeze()
    imag=model.imag_kernels_cache.squeeze()
    print(real.shape,imag.shape)
    z_complex = torch.complex(real,imag)
    fft = torch.fft.fft(z_complex, dim=-1)
    amp = torch.abs(fft).detach().cpu().numpy()
    print(amp.shape)

    im = plt.imshow(amp, cmap="jet", aspect="auto")
    plt.xlabel("Time index (127 points)")
    plt.ylabel("Kernel index (64 filters)")
    plt.title("Amplitude heatmap (64 × 127)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('AWTL-DDFN/fig/kernel.png')
    plt.close()



