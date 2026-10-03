# Alcance y colaboración

Cada integrante trabaja únicamente en su bloque: Dani (1), entorno/datos/parches;
2, OMP/K-SVD/entrenamiento; 3, experimentos/métricas; 4, integración/reporte/presentación.
Esta entrega implementa sólo el bloque 1. No añadir modelos ni resultados ficticios.

Leer docs/00_Plan_general_2_dias.md y el documento del integrante antes de editar.
Respetar contratos, firmas, nombres y señales por columnas. No cambiar contratos
compartidos unilateralmente: describir contradicción y proponer corrección mínima.
src/preprocesamiento.py es la fuente de verdad; el notebook importa ese módulo.
Exportar copias idénticas, nunca mantener implementaciones duplicadas.

CPU, float64, sin clipping en extracción/ensamblado. No GPU ni servicios de pago.
No incluir credenciales ni kaggle.json, ni imprimir secretos. Recursos privados.
No invitar personas sin usuarios exactos y confirmación de destinatarios/permisos.
Mantener main funcional, ramas por integrante y commits pequeños. No force push.
Antes de commit revisar diff y archivos; ejecutar unittest y notebook limpio si
se modifica su ejecución. Distinguir evidencia local de evidencia en Kaggle.
No sobrescribir paquetes aceptados: crear una nueva versión de entrega.
