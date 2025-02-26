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

        // const studentAnswer = await runSegmentScript('segment_model.py', [studentImagePath, 'segments', 'AnswerSheet']);
        // const key = await runOCRPrintedModel('ocr_printed_model.py', [answerKeyPath]);
        const [studentAnswer, key] = await Promise.all([
            runSegmentScript('segment_model.py', [studentImagePath, 'segments', 'AnswerSheet']),
            runOCRPrintedModel('ocr_printed_model.py', [answerKeyPath])
        ]);
        
        const score = await runSimilarityCheckerModel('similarity_checker_model.py', [studentAnswer, key]);
        console.log("going to return")
        return res.json({ data: {score} });
        
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error.' });
    }
});

function runOCRHandwrittenModel(script, args, i, sentences) {
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
            result = Number(data.toString());
        });

        process.stderr.on('data', (data) => {
            console.error(`Error in ${script}:`, data.toString());
        });

        process.on('close', async (code) => {
            if (code !== 0) {
                return reject(`Process exited with code ${code}`);
            }

            if (result === 0) return resolve("");

            let sentences = Array(result).fill("");
            let batchSize = 5;
            let index = 0;

            async function processBatch() {
                if (index >= result) {
                    return resolve(sentences.join(" "));
                }

                let batch = [];
                for (let j = index; j < Math.min(index + batchSize, result); j++) {
                    batch.push(runOCRHandwrittenModel('ocr_handwritten_model.py', [`segments/${args[2]}${j}.png`], j, sentences));
                }

                try {
                    await Promise.all(batch);
                    index += batchSize;
                    processBatch(); // Process next batch
                } catch (error) {
                    reject(error);
                }
            }

            processBatch(); // Start batch processing
        });
    });
}

function runOCRPrintedModel(script, args) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'models', script);
        const absoluteArgs = [path.resolve(args[0])];

        console.log(`Running script: ${scriptPath} with args:`, absoluteArgs);

        const process = spawn('python', [scriptPath, ...absoluteArgs]);

        let result = "";

        process.stdout.on('data', (data) => {
            let text = data.toString();
            console.log("🚀 ~ process.stdout.on ~ text:", text)
            text.slice(0,6) === 'text: ' ? result = text.slice(5) : "";
        });

        process.stderr.on('data', (data) => {
            console.error(`Error in ${script}:`, data.toString());
        });

        process.on('close', (code) => {
            if (code === 0) {
                resolve(result);
            } else {
                reject(`Process exited with code ${code}`);
            }
        });
    });
}
function runSimilarityCheckerModel(script, args) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'models', script);
        const absoluteArgs = [path.resolve(args[0]), path.resolve(args[1])];

        console.log(`Running script: ${scriptPath} with args:`, absoluteArgs);

        const process = spawn('python', [scriptPath, ...absoluteArgs]);

        let result = "";

        process.stdout.on('data', (data) => {
            let text = data.toString();
            console.log("🚀 ~ process.stdout.on ~ text:", text)
            text.slice(0,7) === 'score: ' ? result = text.substring(7, text.length-2) : "";
        });

        process.stderr.on('data', (data) => {
            console.error(`Error in ${script}:`, data.toString());
        });

        process.on('close', (code) => {
            if (code === 0) {
                resolve(result);
            } else {
                reject(`Process exited with code ${code}`);
            }
        });
    });
}
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
