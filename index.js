const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const port = 5000;
app.use(cors());
// Ensure 'uploads' directory exists
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR);
}

// Configure multer to store files with original names & extensions
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOADS_DIR); // Save to 'uploads/' directory
    },
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname); // Get file extension
        const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${ext}`;
        cb(null, uniqueName); // Save file with original extension
    }
});

const upload = multer({ storage: storage });

// Handle image upload (student_image & answer_key)
app.post('/process-images', upload.fields([
    { name: 'student_image', maxCount: 1 },
    { name: 'answer_key', maxCount: 1 }
]), async (req, res) => {
    try {
        if (!req.files || !req.files['student_image'] || !req.files['answer_key']) {
            return res.status(400).json({ error: 'Please upload both student and answer key images.' });
        }

        const studentImagePath = req.files['student_image'][0].path;
        const answerKeyPath = req.files['answer_key'][0].path;

        console.log("Processing uploaded student and answer key images...");

        const studentAnswer = await runSegmentScript('segment_model.py', [studentImagePath, 'segments', 'AnswerSheet']);
        const key = await runSegmentScript('segment_model.py', [answerKeyPath, 'segments', 'AnswerKey']);
        console.log("🚀 ~ ]), ~ reslt:", studentAnswer)
        console.log("going to return")
        return res.json({ data: {studentAnswer, key} });
        
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error.' });
    }
});

function runOCRModel(script, args, i, sentences) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'models', script);
        const absoluteArgs = [path.resolve(args[0])];

        console.log(`Running script: ${scriptPath} with args:`, absoluteArgs);

        const process = spawn('python', [scriptPath, ...absoluteArgs]);

        let result = "";

        process.stdout.on('data', (data) => {
            let text = data.toString();
            console.log("🚀 ~ process.stdout.on ~ text:", text)
            text.slice(0,5) === 'text:' ? result = text.slice(5) : "";
        });

        process.stderr.on('data', (data) => {
            console.error(`Error in ${script}:`, data.toString());
        });

        process.on('close', (code) => {
            if (code === 0) {
                sentences[i]=result;
                resolve(result);
            } else {
                reject(`Process exited with code ${code}`);
            }
        });
    });
}
function runSegmentScript(script, args) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'models', script);
        const absoluteArgs = [path.resolve(args[0]), path.resolve(args[1]), args[2]];

        console.log(`Running script: ${scriptPath} with args:`, absoluteArgs);

        const process = spawn('python', [scriptPath, ...absoluteArgs]);

        let result;

        process.stdout.on('data', (data) => {
            result=Number(data.toString());
        });

        process.stderr.on('data', (data) => {
            console.error(`Error in ${script}:`, data.toString());
        });

        process.on('close', (code) => {
            if (code === 0) {
                if (result == 0) return resolve("");
        
                let sentences = Array(result).fill("");
                let promises = [];
                
                for (let i = 0; i < result; i++) {
                    promises.push(runOCRModel('ocr_model.py', [`segments/${args[2]}${i}.png`], i, sentences));
                }
                
                Promise.all(promises)
                    .then(() => resolve(sentences.join(" ")))
                    .catch(reject); // Ensure rejections are handled
            } else {
                reject(`Process exited with code ${code}`);
            }
        });
        
    });
}

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
