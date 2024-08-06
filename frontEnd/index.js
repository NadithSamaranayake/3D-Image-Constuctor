import multer from 'multer';
import express from 'express';
import { dirname } from "path";
import { fileURLToPath } from "url";
import { exec } from 'child_process';
import fs from 'fs';
import util from 'util';

const execPromise = util.promisify(exec);
const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();
const port = 3000;

app.use(express.static('public'));
app.get("/", (req, res) => {
    res.sendFile(__dirname + "/index2.html");
  });

const storage = multer.diskStorage({
  destination: function(req, file, cb){
    cb(null, 'uploads/');
  },

  filename: function(req, file, cb){
    cb(null, file.originalname)
  }
});

const upload = multer({ storage: storage });

app.post("/upload", upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded.');
  }

  const originalImagePath = `uploads/${req.file.filename}`;
  const processedImagePath = `uploads/processed_${req.file.filename}`;

  try {
    // Execute the Python script to process the image
    await execPromise(`python3 process_image.py ${originalImagePath} ${processedImagePath}`);
    
    // Respond with the URLs of the original and processed images
    res.json({
      originalImageUrl: `/uploads/${req.file.filename}`,
      processedImageUrl: `/uploads/processed_${req.file.filename}`
    });
  } catch (error) {
    res.status(500).send('Error processing image.');
  }
});

app.use('/uploads', express.static('uploads'));

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
