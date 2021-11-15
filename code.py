
#SOUND
#Classe para carregamento de arquivos de som do tipo WAV.
#Objeto iniciado com o caminho de um arquivo WAV que se deseja carregar.
#.....
#Parâmetros
#sound.arquive : Caminho do arquivo
#sound.name : Nome do arquivo
#sound.sprate : Taxa de amostragem
#sound.samples : Vetor de amostras
#sound.length : Tempo de duração
#sound.time : vetor de evolução de tempo
#sound.signal : Vetor de tuplas com amostra e tempo
#sound.fft : Transformada de Fourier do sinal
#sound.fftfreq : vetor eixo das frequências da trasnformada de Fourier
#.....
#Métodos
#sound.play() : Tocar o som
#sound.plot() : Imprime um gráfico do sinal
#sound.plotfft() : Imprime uma visualização do espectro de frequências
#sound.plotfft_pure() : Imprime uma vizualização da transformada de Fourier (Sem módulo)
class sound:
    
    def __init__(self, arquive):
        self.arquive = arquive
        self.name = arquive[arquive.rfind('/')+1:].replace('.wav', '')
        from scipy.io import wavfile
        import numpy as np
        sprate, samples = wavfile.read(arquive)
        self.sprate = sprate
        self.samples = samples
        self.length = len(samples)/sprate
        self.time = np.linspace(0., self.length, len(samples))
        self.signal = list(zip(self.time, samples))
        self.fft = np.fft.fft(samples)
        self.fftfreq = np.fft.fftfreq(len(samples), d=1/sprate)
        return
        
    def play(self):
        from playsound import playsound
        playsound(self.arquive)
        return
        
    def plot(self, size=(20,6), title=None, xzoom=None, yzoom=None):
        import librosa
        import matplotlib.pyplot as plt
        import librosa.display
        import numpy as np 
        plt.figure(figsize=size)
        if title != None:
            plt.title(title)
        plt.xlabel('Tempo')
        plt.ylabel('Amplitude')
        if xzoom != None:
            plt.xlim(0, xzoom)
        if yzoom != None:
            plt.xlim(0, yzoom)
        x = np.array(self.samples)
        #librosa.display.waveshow(x, sr=44100, alpha=0.8)
        librosa.display.waveplot(x, sr=self.sprate, color='orange', alpha=0.8)
        plt.show()
        return 

    def plotfft_pure(self, size=(20,6)):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(1,figsize=size)
        plt.title("FFT")
        plt.xlabel("Frequency(Hz)")
        plt.ylabel("Real")
        plt.plot(self.fftfreq, np.array(self.fft).real, color='darkcyan')
        plt.show()
        return 
    
    def plotfft(self, size=(20,6), xzoom=3000, yzoom=None, lw=1.8):
        import matplotlib.pyplot as plt
        msp = []
        for i in self.fft:
            msp.append(abs(i))
        mfreq = [] 
        for i in self.fftfreq:
            mfreq.append(abs(i))
        plt.figure(1,figsize=size)
        plt.xlabel("Frequência (Hz)")
        plt.xlim(0, xzoom)
        if yzoom != None:
            plt.ylim(0, yzoom)
        plt.ylabel("Módulo")
        plt.plot(mfreq, msp, color='darkcyan', lw=lw)
        plt.show()
        return 

    
    
    
    
#SIGNAL
#Classe para criação de um sinal sonoro.
#Objeto iniciado sem parâmetros.
#.....
#Parâmetros
#signal.sprate : Taxa de amostragem
#signal.samples : Vetor de amostras do sinal criado
#signal.length : Tempo de duração
#signal.time : vetor de evolução de tempo
#signal.fundamental : Valor da frequência fundamental
#.....
#Métodos
#signal.oscilation() : Cria o sinal com a frequência principal pretendida e define o tempo de duração
#signal.add() : Adiciona uma nova frequência ao sinal.
#signal.exp_decay() : Adiciona um decaimento exponencial ao sinal.
#signal.envelopment() : Adiciona um envelope. Requer um objeto da classe envelope
#signal.create() : Cria um arquivo de som WAV com o sinal criado.
#signal.play() : Tocar o som
#signal.plot() : Imprime um gráfico do sinal
#signal.plotfft() : Imprime uma visualização do espectro de frequências  
class signal:
    
    def __init__(self):
        self.samples = None
        self.time = None
        self.length = None
        self.sprate = 44100
        self.fundamental = None
        
    def oscilation(self, f, T):
        import numpy as np
        t = np.linspace(0, T, int(T*self.sprate), endpoint=False)
        x = np.sin(2*np.pi*f*t)
        self.fundamental = f
        self.samples = x
        self.time = t
        self.length = T
        return self
    
    def add(self, f, P):
        import numpy as np
        x = np.sin(2*np.pi*f*self.time)*P
        self.fundamental = f
        self.samples = self.samples + x
        return self
    
    def play(self):
        import numpy as np
        from scipy.io import wavfile
        from playsound import playsound
        X = self.samples*(1/max(self.samples)) #normalização
        wavfile.write('playing.wav', self.sprate, np.float32(X))
        playsound('playing.wav')
        return
    
    def plot(self, size=(20,6), title=None, xzoom=None, yzoom=None):
        import librosa
        import matplotlib.pyplot as plt
        import librosa.display
        import numpy as np 
        plt.figure(figsize=size)
        if title != None:
            plt.title(title)
        plt.xlabel('Tempo')
        plt.ylabel('Amplitude')
        if xzoom != None:
            plt.xlim(0, xzoom)
        if yzoom != None:
            plt.xlim(0, yzoom)
        x = np.array(self.samples)
        #librosa.display.waveshow(x, sr=self.sprate, alpha=0.8)
        librosa.display.waveplot(x, sr=self.sprate, color='orange', alpha=0.8)
        plt.show()
        return
    
    def plotfft(self, size=(20,6), xzoom=3000, yzoom=None, lw=1.8):
        import matplotlib.pyplot as plt
        import numpy as np
        self.fft = np.fft.fft(self.samples)
        self.fftfreq = np.fft.fftfreq(len(self.samples), d=1/self.sprate)
        msp = []
        for i in self.fft:
            msp.append(abs(i))
        mfreq = [] 
        for i in self.fftfreq:
            mfreq.append(abs(i))
        plt.figure(1,figsize=size)
        #plt.title("Espectro")
        plt.xlabel("Frequência (Hz)")
        plt.xlim(0, xzoom)
        if yzoom != None:
            plt.ylim(-(yzoom/50), yzoom)
        plt.ylabel("Módulo")
        plt.plot(mfreq, msp, color='darkcyan', lw=lw)
        plt.show()
        return 
        
    def exp_decay(self, alpha):
        import numpy as np
        dt = 1/self.sprate
        t = np.arange(0, self.length, dt)
        self.samples = self.samples*np.exp(alpha*(self.length - t))*120
        return self
    
    def envelopment(self, env):
        import numpy as np
        L = min([len(self.samples), len(env.samples)])
        t=self.time[:L] ; sp=self.samples[:L] ; env=env.samples[:L]
        newsp = np.array(sp)*np.array(env)
        self.samples = newsp
        return self
    
    def create(self, nome):
        import numpy as np
        from scipy.io import wavfile
        #Normalizar antes de criar o arquivo
        X = self.samples*(1/max(self.samples))
        wavfile.write(nome+'.wav', self.sprate, np.float32(X))
        return nome+'.wav'
    
    
    
    
    
    
#ENVELOPE
#Classe para criação, coleta e tratamento de envelope
#Objeto iniciado sem parâmetros
#.....
#Parâmetros
#envelope.samples : Vetor de amostras do envelope
#envelope.lenght : Comprimento do envelope
#envelope.sprate : Taxa de amostragem
#.....
#Métodos
#envelope.get_hilbert() : Obtém o envelope de um som através do método de Hilbert
#envelope.lowpass() : Filtro passa baixa para o envelope
#envelope.geometric() : Cria um envelope geométrico
class envelope:

    def __init__(self):
        self.samples = None
        self.length = None
        self.sprate = 44100
        return
        
    def get_hilbert(self, sig):
        from scipy.signal import hilbert
        import numpy as np
        
        s = hilbert(sig.samples)
        self.samples = np.abs(s)
        self.lenght = len(s)
        return self
    
    def plot(self, size=(26,6), color="brown", lw=1):
        import matplotlib.pyplot as plt
        if type(self.samples) == type(None):
            print("Empty envelope!")
        else:
            plt.figure(figsize=size)
            plt.plot(self.samples, color=color, lw=lw)
            plt.xticks([])  
            plt.yticks([])
            plt.show()
        return
    
    def exp_decay(self, alpha, length):
        import numpy as np
        dt = 1/self.sprate
        t = np.arange(0, length, dt)
        if self.samples == None:
            self.samples = np.exp(alpha*(length - t))*120
        else:
            self.samples = self.samples*np.exp(alpha*(self.length - t))*120
        return self
    
    def geometric(self, lar, length):
        import numpy as np
        dt = 1/self.sprate
        intv = []
        coefs = []
        for i in range(1,len(lar)):
            intv.append((lar[i-1][0],lar[i][0]))
            if (lar[i][0] - lar[i-1][0]) == 0:
                coefs.append(1)
            else:
                coefs.append((lar[i][1] - lar[i-1][1])/(lar[i][0] - lar[i-1][0]))
        intv.append((lar[i][0],length))
        if (length - lar[i][0]) == 0:
            coefs.append(1)
        else:
            coefs.append((1 - lar[i-1][1])/(length - lar[i][0]))
        env = []
        for i, c, l in zip(intv, coefs, lar):
            env = env + list(l[1] + c*np.arange(0, i[1]-i[0],dt))
        self.samples = env[:int(self.sprate*length)]
        self.length = length
        return self
    
    def lowpass(self, wd, N=2):
        from scipy import signal
        b, a = signal.butter(N, wd, btype='lowpass')
        self.samples = signal.filtfilt(b, a, self.samples)
        return self
    
    def highpass(self, wd, N=2):
        from scipy import signal
        b, a = signal.butter(N, wd, btype='highpass')
        self.samples = signal.filtfilt(b, a, self.samples)
        return self
    
    
    
    
    
#SPECTRUM
#Classe para criação, coleta e tratamento de espectros de som
#Objeto iniciado sem parâmetros
#.....
#Parâmetros
#spectrum.samples : Vetor de amostras do espectro
#spectrum.freqs : Lista de frequências que compõem o espectro
#spectrum.sprate : Taxa de amostragem
#spectrum.fundamental : Valor da frequência fundamental do espectro
#spectrum.compare : parametro de comparação
#spectrum.principls : Vetor de frequencias principais ordenadas e suas proporções
#.....
#Métodos
#spectrum.get() : obtém o espectro de um som passado e porcentagem de consideração de frequencias
#spectrum.plot() : Mostra o espectro e o limite de consideração de frequencias
class spectrum:
    
    def __init__(self):
        self.samples = None
        self.freqs = None
        self.sprate = None
        self.fundamental = None
        self.compare = None
        self.principals = None
        return
        
    def get(self, sound, accuracy):
        import numpy as np
        self.samples = np.fft.fft(sound.samples)
        self.freqs = np.fft.fftfreq(len(sound.samples), d=1/sound.sprate)
        self.sprate = sound.sprate
        sig = []
        for i in range(len(self.freqs)):
            sig.append((abs(self.samples[i]), abs(self.freqs[i])))  
        topos = []
        self.fundamental = max(sig)[1]
        self.compare = (accuracy/100)*max(sig)[0]
        for i in range(len(sig)):
            if i < len(sig) - 2:
                if sig[i][0] > sig[i-1][0] and sig[i][0] > sig[i+1][0] and sig[i][0] > self.compare:
                    topos.append(sig[i])
        topos = topos[:int(len(topos)/2)]
        topos.sort(reverse = True)
        self.principals = topos
        return self
    
    def plot_pure(self, size=(20,6)):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(1,figsize=size)
        plt.title("FFT")
        plt.xlabel("Frequency(Hz)")
        plt.ylabel("Real")
        plt.plot(self.freqs, np.array(self.samples).real, color='darkcyan')
        plt.show()
        return 
    
    def plot(self, size=(20,6), xzoom=3000, yzoom=None, lw=1.8):
        import matplotlib.pyplot as plt
        msp = []
        for i in self.samples:
            msp.append(abs(i))
        mfreq = [] 
        for i in self.freqs:
            mfreq.append(abs(i))
        plt.figure(1,figsize=size)
        #plt.title("Espectro")
        plt.xlabel("Frequência (Hz)")
        plt.xlim(0, xzoom)
        if yzoom != None:
            plt.ylim(0, yzoom)
        plt.ylabel("Módulo")
        plt.plot(mfreq, msp, color='darkcyan', lw=lw)
        if self.compare != None and self.principals != None:
            plt.axhline(y=self.compare, color='red')
            for i in self.principals:
                plt.scatter(i[1], i[0], color='red')
        plt.show()
        return
        
        
        
        
#PULSES_DATABASE
#Classe da base de dados
class pulses_database:
    
    def __init__(self, path = '/home/dakkar/Graduação Fisica Computacional USP/TCC/pulse.data'):
        
        from os import listdir
        from os.path import isfile, isdir, join
        
        subpaths = [f for f in listdir(path) if isdir(join(path, f))]
        self.instruments = [f.replace('.data', '') for f in subpaths]
        self.objects = dict()
        for i in self.instruments:
            self.objects[i] = path+'/'+i+'.data'
        return
    
    def dataset(self, inst):
        from os import listdir
        from os.path import isfile, isdir, join
        path = self.objects[inst]
        files = [path+'/'+f for f in listdir(path) if isfile(join(path, f))]
        dataset = dict()
        for i in files:
            key = i.replace(path+'/','').replace(inst+'_','').replace('.wav','')
            dataset[key] = i
        return dataset
    
    def show(self):
        print('Database Available Instruments and Their Respective Keys \n')
        for i in self.instruments:
            print('\t'+i, end=': ')
            print([i for i in self.dataset(i).keys()], end='\n\n')

            
            
            
#Outros métodos

def sintet(som, percent=1.0):
    spec = spectrum()
    spec.get(som, percent)
    sig = signal()
    sig.oscilation(spec.principals[0][1], 4)
    for i in spec.principals[:nh]:
        proportion = i[0]/spec.principals[0][0]
        sig.add(i[1], proportion)
    return sig

def invert(a, b, nh=50, filtparam=0.002):
    spec = spectrum()
    spec.get(a, 1)
    env = envelope()
    env.get_hilbert(b).lowpass(filtparam)
    env.plot()
    sig = signal()
    sig.oscilation(spec.principals[0][1], 4)
    for i in spec.principals[:nh]:
        proportion = i[0]/spec.principals[0][0]
        sig.add(i[1], proportion)
    sig.envelopment(env)
    return sig
