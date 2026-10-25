# Yatzhee-aplicando-metodo-montecarlo
Juego de Yatzhee aplicando el metodo montecarlo, realizado en python para la clase de simulacion del programa de ingenieria de Software de la Institucion Universitaria Digital de Antioquia

Descripción del Proyecto

Este proyecto implementa una simulación computacional del clásico juego de dados Yahtzee para dos jugadores. El objetivo central es construir un motor de toma de decisiones automatizado apoyado en el Método de Montecarlo. Dado que Yahtzee tiene millones de combinaciones probables, es complejo calcular analíticamente qué dados retener en cada turno. Nuestro algoritmo genera escenarios simulados rápidos (explorando las 32 máscaras de bloqueo posibles para 5 dados) y selecciona la jugada que maximiza el Valor Esperado (Expected Value - EV).

Sobre el Código

El código está escrito en Python y se estructura modularmente. Utiliza el módulo random para la generación estocástica de lanzamientos y aplica dataclasses para mantener un estado limpio de los jugadores y de las estadísticas del juego. Además, integra librerías de análisis de datos (Pandas, Matplotlib, Seaborn) para validar visualmente la convergencia de la distribución de probabilidad (demostrando la Teoría de los Grandes Números).
