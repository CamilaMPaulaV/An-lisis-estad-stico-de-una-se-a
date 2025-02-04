# Statistical analysis of a signal
## Introdución
El siguiente código contiene todo lo requerido para graficar una señal ECG, sacar sus datos estadísticos tanto de manera manual como automática, además de realizar su histograma y función de probabilidad correspondiente. Así mismo, se contamina la señal con tres tipos de ruidos específicos para sacar su SNR 

## Procedimiento
Primero se descargó una derivación de una señal ECG mediante la biblioteca de Pshyonet en archivo .mat, la anterior será la señal base. 

Una vez realizado lo anterior se gráfico (observelo en la fig 1) y se realizó su media, desviación estandandar, coeficiente de variación, histograma y función de probabilidad, para verificar la certeza de lo anterior se realizó de forma manual y automática (las que permitia python). Añadiendo se contaminó la señal original con ruido para simular efectos que podrían pasar en la vida real y enconrrar su realación señal ruido (SNR)
