import multer from "multer";
import express from "express";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

//setting up directory name for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

app.get("/", (req, res) => {
    res.sendFile(__dirname + "/index4.html");
  });

//Setting up storage for uploaded images
const storage = multer.diskStorage({
    destination: './uploads',
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage });

//serve static files
app.use(express.static(path.join(__dirname, 'public')));

//upload and process image
app.post('/process', upload.single('image'), (req, res) => {
    const imagePath = path.join(__dirname, 'uploads', req.file.filename);
    const outputPath = path.join(__dirname, 'uploads', 'processed', req.file.filename);
    const filterType = req.body.filterType;   

    //Execute python code
    const pythonProcess = spawn('python3', ['process_image.py', imagePath, outputPath, filterType]);

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.json({ processedImageUrl: '/uploads/processed_' + req.file.filename });
        } else {
            res.status(500).send('Image processing failed.');
        }
    });
});

//start the server
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});