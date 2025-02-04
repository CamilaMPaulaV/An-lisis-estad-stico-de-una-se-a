# Statistical analysis of a signal
## Introdución
El siguiente código contiene todo lo requerido para graficar una señal ECG, sacar sus datos estadísticos tanto de manera manual como automática, además de realizar su histograma y función de probabilidad correspondiente. Así mismo, se contamina la señal con tres tipos de ruidos específicos para sacar su SNR 

## RESULTADOS
Primero se descargó la derivación (¿?) de una señal ECG mediante la biblioteca de Pshyonet en archivo .mat, la anterior será la señal base. Una vez realizado lo anterior se obtuvo la siguiente gráfica
AQUÍ LA GRÁFICA

Posteriormente se realizó su media, desviación estandandar y coeficiente de variación para los cuales se obtuvo lo siguiente:

Media automática: 0.009123999999999998  
Media manual: 0.009124

Desviación estándar automática: 0.13012802397639028  
Desviación estándar manual: 0.13014104

Coeficiente de variación auomático: no tiene función predeterminada  
Coeficiente de variación manual: 1426.35947754

Una vez obtenidos los datos anteriores se realizó su histograma y su función de probabilidad de manera manual y automática obteniendo lo siguiente:
AQUÍ LAS DOS GRÁFICAS 

Añadiendo se contaminó la señal original con ruido para simular efectos que podrían pasar en la vida real y encontrar su realación señal ruido (SNR). Para lo siguiente es necesario tener
en cuenta que el SNR hace referencia a la diferencia que existe entre el ruido y la potencia de una señal, lo que quiere decir que mide la proporción entre la señal útil y el ruido
presente, teniendo en cuenta lo anterior es importante decir que un mayor SNR indica una señal más clara, ya que la potencia de la señal es considerablemente mayor que la del ruido, lo
que mejora su calidad. De esta manera se obtuvo:
AQUÍ LAS GRÁFICAS

SNR con Ruido Gaussiano: -0.23 dB
SNR con Ruido Impulso: -18.66 dB
SNR con Ruido Artefacto: -0.84 dB

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

3. A continuación se realiza la impresión en la consola de los datos sacados anteriormente, puede hacerlo despues de cada función sacada o al final (como en este caso). Se usa la libreria matplotlib y seaborn.
   
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
plt.ylabel('Frecuencia (Hz)')
plt.show()

plt.bar(bin_bor[:-1], frecuencia, width=(bin_bor[1] - bin_bor[0]), color='pink', edgecolor='black', align='edge')
plt.title('Histograma Manual de la Señal ECG')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Frecuencia (Hz)')
plt.show()

sns.kdeplot(ecg, color='green')
plt.title('Función de Probabilidad con Funciones de Python')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Densidad de probabilidad')
plt.show()

sns.kdeplot(ecg, color='green')
plt.title('Función de Probabilidad manual')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Densidad de probabilidad')
plt.show()
```
4. A continuación se realiza la parte del SNR, contaminando la señal con tres tipos de ruido. Primero se declara una función retornable llamada calcular_snr en la cual se determina la fórmula que usa para sacar el mismo. Posterioemnte se realiza el rudio gaussiano que se realiza entorno a los valores de la media y de la desviacion estandar

```python
#SNR
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal**2)
    potencia_ruido = np.mean(ruido**2)
    snr = 10 * np.log10(potencia_señal / potencia_ruido)
    return snr

# Ruido Gaussiano
ruido_gaussiano = np.random.normal(media, desviacion_estandar) 
ecg_gaussiano = ecg + ruido_gaussiano
snr_gaussiano = calcular_snr(ecg, ruido_gaussiano)

#Ruido Impulso
ruido_impulso = np.zeros_like(ecg)
indices_impulso = np.random.choice(np.arange(len(ecg)), size=int(len(ecg) * 0.05), replace=False)
ruido_impulso[indices_impulso] = np.random.choice([-5, 5], size=indices_impulso.shape)  #Picos de ±5
ecg_impulso = ecg + ruido_impulso
snr_impulso = calcular_snr(ecg, ruido_impulso)

#Ruido Artefacto 
ruido_artefacto = np.random.normal(0, 0.2, ecg.shape) * np.sin(2 * np.pi * 1 * t)  #Por una señal senoidal
ecg_artefacto = ecg + ruido_artefacto
snr_artefacto = calcular_snr(ecg, ruido_artefacto)

#Graficar las señales contaminadas
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, ecg_gaussiano)
plt.title(f'Señal ECG con Ruido Gaussiano (SNR = {snr_gaussiano:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.subplot(3, 1, 2)
plt.plot(t, ecg_impulso)
plt.title(f'Señal ECG con Ruido Impulso (SNR = {snr_impulso:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.subplot(3, 1, 3)
plt.plot(t, ecg_artefacto)
plt.title(f'Señal ECG con Ruido Artefacto (SNR = {snr_artefacto:.2f} dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')

plt.tight_layout()
plt.show()

print(f"\nSNR con Ruido Gaussiano: {snr_gaussiano:.2f} dB")
print(f"SNR con Ruido Impulso: {snr_impulso:.2f} dB")
print(f"SNR con Ruido Artefacto: {snr_artefacto:.2f} dB")
```
   
