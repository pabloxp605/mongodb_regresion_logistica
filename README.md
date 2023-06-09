# mongodb_regresion
Trabajo Final Inteligencia Negocios - MongoDB +Regresion Logistica


## Entorno de trabajo

 - Sistema Operativo Linux Ubuntu 22
 - Version de R version 4.3.0 
 - Libreria para R de mongo es mongolite


## Instrucciones
 - Levantar el docker de mongoDB en la carpeta mongo con el comando 
 ```
 docker-compose up
 ```

 - Conectar mongo con R

 Ejemplo de conexion a una base de datos PG y una collection productos

### Establecer la conexion
con <- mongo(url = "mongodb://localhost:27017/pg")
con <- mongo(collection = "productos", db="pg", url = "mongodb://localhost:27017/pg")

### Establecer una conexion OK
con <- mongo(collection = "productos", db="pg", url = "mongodb://localhost:27017/pg")
											
### example insert
con$insert(iris)

### example query
result <- con$find()
result <- con$find('{"Species" : "setosa"}')