# Statistical analysis of a signal
## Introducción
El siguiente código contiene todo lo requerido para graficar una señal ECG, sacar sus datos estadísticos tanto de manera manual como automática, además de realizar su histograma y función de probabilidad correspondiente. Así mismo, se contamina la señal con tres tipos de ruido específicos para sacar su SNR tanto para un valor de frecuencia baja como para un valor de frecuencia alta.

## Resultados
Primero se descargó la derivación DI la cual registra la diferencia de potencial entre brazo derecho y brazo izquierdo de una señal ECG. La descarga se realizó mediante la biblioteca de Pshyonet en archivo .mat, obteniendo así la que será la señal base. Una vez realizado lo anterior se obtuvo la siguiente gráfica

<div align="center">
  <img src="https://github.com/user-attachments/assets/0f9d9e53-52d2-4e14-b154-2297028645f3" width="400" height="300">
</div>

Posteriormente se realizaron los cálculos pertinentes para hallar su media, desviación estandandar y coeficiente de variación, los valores obtenidos son los siguientes:

Media automática: 0.009123999999999998  
Media manual: 0.009124

Desviación estándar automática: 0.13012802397639028  
Desviación estándar manual: 0.13014104

Coeficiente de variación auomático: no tiene función predeterminada  
Coeficiente de variación manual: 1426.35947754

Una vez obtenidos los datos anteriores se realizó su histograma y su función de probabilidad de manera manual y automática obteniendo lo siguiente:

<div align="center">
<img src="https://github.com/user-attachments/assets/07d92939-5102-4892-9fec-3db30a51b156" width="400" height="300">
<img src="https://github.com/user-attachments/assets/7e1eab35-f89b-4408-8340-17e7a54657d9" width="400" height="300">
<img src="https://github.com/user-attachments/assets/11be7eb3-af39-4c89-a9a1-c128f336f34c" width="400" height="300">
<img src="https://github.com/user-attachments/assets/98aa6348-0345-4532-817d-d54a0a3dfc8d" width="400" height="300">
</div>

Además se contaminó la señal original con ruido a altas y bajas frecuencias para simular efectos que podrían pasar en la vida real y encontrar su realación señal ruido (SNR), entre estos encontramos, el ruido Gaussiano que es un ruido estadistico aleatorio con una distribución de probabilidad normal, el ruido de impulso genera picos repentinos y de alta amplitud que aparecen repentinamente y el ruido artefacto son distorsiones dadas por el sistema, por ejemplo durante el muestreo. 

Para lo siguiente es necesario tener en cuenta que el SNR hace referencia a la diferencia que existe entre el ruido y la potencia de una señal, lo que quiere decir que mide la proporción entre la señal útil y el ruido presente, teniendo en cuenta lo anterior es importante decir que un mayor SNR indica una señal más clara, ya que la potencia de la señal es considerablemente mayor que la del ruido, lo que mejora su calidad.

<div align="center">
  <img src="https://github.com/user-attachments/assets/67a83712-2f54-420c-bf4b-236f16089d5f" width="600" height="500">
   <img src="https://github.com/user-attachments/assets/044a2f5e-2969-420d-8f73-17dd21908e71" width="600" height="500">
</div>



SNR Gaussiano Baja Frecuencia: 2.96 dB   
SNR Gaussiano Alta Frecuencia: -10.46 dB  

SNR Impulso Baja Frecuencia: -18.66 dB    
SNR Impulso Alta Frecuencia: -24.68 dB  

SNR Artefacto Baja Frecuencia: -0.81 dB   
SNR Artefacto Alta Frecuencia: -0.60 dB   

A partir de los anteriores datos se logró evidenciar que el ruido gaussiano afecta principalmente en altas frecuencias, dado que la relación señal-ruido es muy negativa, el ruido impulsivo tiene una relación SNR muy baja tanto en baja como en alta frecuencia Y los ruidos artefactos afectan en menor medida tanto en baja como en alta frecuencia.

## Requerimientos 
1. Python 3.12
2. Librerias Numpy, matplotlib, scipy.io, seaborn y scipy.signal
3. Señal biomédica en formato .mat

## Instrucciones
1. Para cargar la señal debe colocar su ruta del computador donde se encuentra el archivo con el ECG, se reduce la frecuencia para poder ver de forma más específica la señal (se divide en 200, usted puede escoger el valor acorde a sus necesidades), posteriormente se cuentan los datos. Se definene la frecuencia de muestreo y su periodo  y se guardan los datos en un array. Para graficar la señal se usa matplot
   
```python
#CARGAR Y PROCESAR LA SEÑAL ECG
x = loadmat('C:/Users/usuario/Desktop/rec_1m.mat') 
ecg = (x['val']-0)/200
ecg = np.transpose(ecg)

fs = 500  
ts = 1/fs 
t = np.linspace(0, np.size(ecg), np.size(ecg)) * ts 

#GRAFICAR EL ECG
plt.plot(t,ecg)
plt.title('Señal de ECG')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.show()
```
2. Se realizan los cálculos manuales mediante las fórmulas estadísticas determinadas, para lo anterior se realizan bucles para recorrer todos los valores del arreglo, a su vez se realizan de manera automátcia con la libreria numpy en el caso de la media y la desviación estandar, el coeficiente de variación no tiene función prederterminada. Una vez realizado lo anterior se realiza el histograma determinando la cantidad de divisines que tendrá (bins) y su conversión analoga digital (la frecuencia correspondiente a cada bins). El histograma automático también se realiza con numpy
```python
#CÁLCULOS MANUALES Y AUTOMÁTICOS
n = ecg.shape[0]
suma = 0
suma_cuadrados = 0

for i in range(n):
    suma += ecg[i]
media_man = suma / n
media = np.mean(ecg) #automático

for i in range(n):
    suma_cuadrados += (ecg[i] - media_man) **2
desviacion_man = (suma_cuadrados /( n - 1)) **0.5
desviacion_estandar = np.std(ecg) #automático

coeficiente_man = (desviacion_man / media_man) * 100
coeficiente_variacion = (desviacion_estandar / media) * 100

#HISTOGRAMA MANUAL
ecg = ecg.flatten()
num_bins = 60
val_min = min(ecg)
val_max = max(ecg)
anch_bin = (val_max - val_min) / num_bins
frecuencia = [0] * num_bins 

for i in range(n):
    bin_dat = int((ecg[i] - val_min) / anch_bin)
    if bin_dat == num_bins:
        bin_dat -= 1
    frecuencia [bin_dat] += 1
    
bin_bor = [val_min + i * anch_bin for i in range(num_bins)]
probabilidad = [f / n for f in frecuencia]

#HISTOGRAMA AUTOMÁTICO
frecuencia, bin_bor = np.histogram(ecg, bins=60, density=True)
```

3. A continuación se realiza la impresión en la consola de los datos sacados anteriormente, puede hacerlo después de cada función o al final (como en este caso). Se usa la libreria matplotlib y seaborn.
   
```python

#ESCRIBE HISTOGRAMAS Y CÁLCULOS
print("Media manual:", media_man)
print("Desviación estándar manual:", desviacion_man)
print("Coeficiente de variación manual:", coeficiente_man, "\n") 

print("Media:", media)
print("Desviación estándar:", desviacion_estandar)
print("Coeficiente de variación:", coeficiente_variacion)

plt.hist(ecg, bins=60, color='pink', edgecolor='black')
plt.title('Histograma de la señal ECG')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Número de muestras')
plt.show()

plt.bar(bin_bor[:-1], frecuencia, width=(bin_bor[1] - bin_bor[0]), color='pink', edgecolor='black', align='edge')
plt.title('Histograma Manual de la Señal ECG')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Número de muestas')
plt.show()

sns.kdeplot(ecg, color='green')
plt.title('Función de Probabilidad con Funciones de Python')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Densidad de probabilidad')
plt.show()

sns.kdeplot(ecg, color='green')
plt.title('Función de Probabilidad Manual')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Densidad de probabilidad')
plt.show()

```
4. Se da inicio a la parte del SNR, contaminando la señal con tres tipos de ruido. Primero se declara una función retornable llamada calcular_snr en la cual se determina la fórmula que usa para obtener el dato correspondiente, así mismo se calcula la frecuencia de nyquist y de corte que permiten crear el filtro pasa bajo y pasa alto con ayuda de las funciones butter y filtfilt. Posteriormnte se establecen el rudio gaussiano, el ruido de impulso y el ruido artefacto acorde a sus fórmulas permitiendo así calcular el SNR con las distintas frecuencias.

```python
#SNR
def calcular_snr(señal, ruido):

    potencia_señal = np.mean(señal ** 2)
    potencia_ruido = np.mean(ruido ** 2)
    snr = 10 * np.log10(potencia_señal / potencia_ruido)
    return snr

def butter_lowpass_filter(data, frcor, fs, order=4):
    nyq = 0.5 * fs
    normal_frcor = frcor / nyq
    b, a = butter(order, normal_frcor, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, frcor, fs, order=4):
    nyq = 0.5 * fs
    normal_frcor = frcor / nyq
    b, a = butter(order, normal_frcor, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

#RUIDO GAUSSIANO
# Generar ruido blanco gaussiano
ruido_gaussiano = np.random.normal(0, 0.5, ecg.shape)

# Baja frecuencia: filtrar el ruido para obtener componentes por debajo de 10 Hz
ruido_gaussiano_baja = butter_lowpass_filter(ruido_gaussiano, frcor=10, fs=fs)

# Alta frecuencia: filtrar el ruido para obtener componentes por encima de 50 Hz
ruido_gaussiano_alta = butter_highpass_filter(ruido_gaussiano, frcor=50, fs=fs)

ecg_gaussiano_baja = ecg + ruido_gaussiano_baja
ecg_gaussiano_alta = ecg + ruido_gaussiano_alta

snr_gaussiano_baja = calcular_snr(ecg, ruido_gaussiano_baja)
snr_gaussiano_alta = calcular_snr(ecg, ruido_gaussiano_alta)

# RUIDO IMPULSO
# Baja frecuencia: menor cantidad de impulsos (5% de las muestras)
indices_impulso_baja = np.random.choice(np.arange(n), size=int(n * 0.05), replace=False)
ruido_impulso_baja = np.zeros_like(ecg)
ruido_impulso_baja[indices_impulso_baja] = np.random.choice([-5, 5], size=indices_impulso_baja.shape)

# Alta frecuencia: mayor cantidad de impulsos (20% de las muestras)
indices_impulso_alta = np.random.choice(np.arange(n), size=int(n * 0.20), replace=False)
ruido_impulso_alta = np.zeros_like(ecg)
ruido_impulso_alta[indices_impulso_alta] = np.random.choice([-5, 5], size=indices_impulso_alta.shape)

ecg_impulso_baja = ecg + ruido_impulso_baja
ecg_impulso_alta = ecg + ruido_impulso_alta

snr_impulso_baja = calcular_snr(ecg, ruido_impulso_baja)
snr_impulso_alta = calcular_snr(ecg, ruido_impulso_alta)

# RUIDO ARTEFACTO
# Baja frecuencia: se modula el ruido con una sinusoide de 1 Hz
ruido_artefacto_baja = np.random.normal(0, 0.2, ecg.shape) * np.sin(2 * np.pi * 1 * t)
ecg_artefacto_baja = ecg + ruido_artefacto_baja
snr_artefacto_baja = calcular_snr(ecg, ruido_artefacto_baja)

# Alta frecuencia: se modula el ruido con una sinusoide de 50 Hz
ruido_artefacto_alta = np.random.normal(0, 0.2, ecg.shape) * np.sin(2 * np.pi * 50 * t)
ecg_artefacto_alta = ecg + ruido_artefacto_alta
snr_artefacto_alta = calcular_snr(ecg, ruido_artefacto_alta)
```
5. Se permite la visualización de las señales contamionadas mediante matplotlib de los tres ruidos con sus respectivas frecuencias.
```python
   
# Señales contaminadas con ruidos de BAJA FRECUENCIA
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(t, ecg_gaussiano_baja)
plt.title(f'ECG con Ruido Gaussiano Baja Frecuencia (SNR = {snr_gaussiano_baja:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.subplot(3, 1, 2)
plt.plot(t, ecg_impulso_baja)
plt.title(f'ECG con Ruido Impulso Baja Frecuencia (SNR = {snr_impulso_baja:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.subplot(3, 1, 3)
plt.plot(t, ecg_artefacto_baja)
plt.title(f'ECG con Ruido Artefacto Baja Frecuencia (SNR = {snr_artefacto_baja:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.tight_layout()
plt.show()

# Señales contaminadas con ruidos de ALTA FRECUENCIA
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(t, ecg_gaussiano_alta)
plt.title(f'ECG con Ruido Gaussiano Alta Frecuencia (SNR = {snr_gaussiano_alta:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.subplot(3, 1, 2)
plt.plot(t, ecg_impulso_alta)
plt.title(f'ECG con Ruido Impulso Alta Frecuencia (SNR = {snr_impulso_alta:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.subplot(3, 1, 3)
plt.plot(t, ecg_artefacto_alta)
plt.title(f'ECG con Ruido Artefacto Alta Frecuencia (SNR = {snr_artefacto_alta:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.tight_layout()
plt.show()

print(f"\nSNR Gaussiano Baja Frecuencia: {snr_gaussiano_baja:.2f} dB")
print(f"SNR Gaussiano Alta Frecuencia: {snr_gaussiano_alta:.2f} dB")
print(f"SNR Impulso Baja Frecuencia: {snr_impulso_baja:.2f} dB")
print(f"SNR Impulso Alta Frecuencia: {snr_impulso_alta:.2f} dB")
print(f"SNR Artefacto Baja Frecuencia: {snr_artefacto_baja:.2f} dB")
print(f"SNR Artefacto Alta Frecuencia: {snr_artefacto_alta:.2f} dB") 
```
## Uso
Statistical analysis of a signal by Camila Martínez and Paula Vega  
Published 4/02/25

