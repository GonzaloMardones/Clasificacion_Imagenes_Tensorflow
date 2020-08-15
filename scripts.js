let net
const webcamElement = document.getElementById("webcam")
const classifier = knnClassifier.create()


const app = async () => {
    console.log("Cargando modelo de identificacion de imagenes");
    net = await mobilenet.load()
    console.log("Carga terminada")
    webcam = await tf.data.webcam(webcamElement)

    while(true){
        const img = await webcam.capture()
        const result = await net.classify(img)
        displayImagePrediction(img)

        const activacion =  net.infer(img, "conv_preds") //tomo la salida anterior y utilizo una red convolucional de mobilnet
        // se genera un tensor de activacion (aun no nos dice si es un perro o lo que sea)
        
        var resultado
        try{
            resultado = await classifier.predictClass(activacion) //predice que clase es en base a la activacion de la red anterior
            
        }catch(error){
            resultado={}
        }

        const classes = ["Inexperto","Pared","Persona","OK","Taza","Telefono"]

        //retorno a la web
        document.getElementById('modelo').innerHTML = `
        Predicción: ${result[0].className} <br>
        Probabilidad:${result[0].probability}`

        try {
            document.getElementById("modelo_knn").innerText = `
          Predicción: ${classes[resultado.label]}
          Probabilidad: ${resultado.confidences[resultado.label]}`
          } catch (error) {
            document.getElementById("modelo_knn").innerText="Inexperto (Sin experiencia)";
          }
              
        img.dispose()
        await tf.nextFrame()
    }

} 

//  Usamos un modelo de tipo mobilenet, pero podemos transferir el conocimiento con un clasificador knn
//  y para esto clasificamos las imagenes que tenemos y entrenaremos el modelo con las imagenes obtenidas
const addExample = async (classId) =>{
    console.log('add Example')
    const img = await webcam.capture()
    const activacion = net.infer(img,true)
    classifier.addExample(activacion, classId)

    img.dispose()
}

const displayImagePrediction = async (imgElement) => {
    try{
        result = await net.classify(imgElement)

    }catch(error){
        console.log('Error, de tipo: ',error.toString())
    }
}

//Guardamos el dataset del clasificador (etiquetas y vectores)
const saveKnn = async () => {    
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    const storageKey = "knnClassifier";
    //Almacenamos en localStorage
    localStorage.setItem(storageKey, strClassifier);
};

//Cargamos nuestro modelo de localStorage
const loadKnn = async ()=>{
    const storageKey = "knnClassifier";
    let datasetJson = localStorage.getItem(storageKey);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};


//Se ejecuta el script principal
app()
 