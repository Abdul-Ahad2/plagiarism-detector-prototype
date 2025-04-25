const express = require("express");
const axios = require("axios");
const cheerio = require("cheerio");
const multer = require("multer");
const cors = require("cors");
const { PythonShell } = require("python-shell");
const pdf = require("pdf-parse");
const fs = require("fs");
const path = require("path");

const app = express();
const port = 4000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer to handle both file and text fields
const upload = multer({ storage: multer.memoryStorage() });

let scrapedCorpus = [];

async function initializeCorpus() {
  console.log(
    `[${new Date().toISOString()}] [INFO] Fetching Wikipedia page content for "Plagiarism"`
  );
  try {
    const response = await axios.get(
      "https://en.wikipedia.org/wiki/Plagiarism"
    );
    const $ = cheerio.load(response.data);
    const content = $("#mw-content-text .mw-parser-output p")
      .map((i, el) => $(el).text().trim())
      .get()
      .join(" ");
    scrapedCorpus = [content];
    console.log(
      `[${new Date().toISOString()}] [INFO] Wikipedia content fetched. Length: ${
        content.length
      } characters`
    );
    console.log(
      `[${new Date().toISOString()}] [INFO] Corpus initialized with ${
        scrapedCorpus.length
      } text(s)`
    );
  } catch (error) {
    console.error(
      `[${new Date().toISOString()}] [ERROR] Failed to fetch Wikipedia content: ${
        error.message
      }`
    );
    scrapedCorpus = [];
  }
}

initializeCorpus();

app.post(
  "/check-plagiarism",
  upload.fields([{ name: "file", maxCount: 1 }, { name: "text" }]),
  async (req, res) => {
    console.log(
      `[${new Date().toISOString()}] [INFO] Received /check-plagiarism request`
    );
    console.log(
      `[${new Date().toISOString()}] [DEBUG] Request body: ${JSON.stringify(
        req.body
      )}`
    );
    console.log(
      `[${new Date().toISOString()}] [DEBUG] Request files: ${
        req.files?.file ? req.files.file[0].originalname : "none"
      }`
    );
    console.log(
      `[${new Date().toISOString()}] [DEBUG] Content-Type: ${req.get(
        "Content-Type"
      )}`
    );

    let inputText = req.body.text || "";

    // Extract text from file if provided and no text input
    if (!inputText && req.files?.file) {
      const file = req.files.file[0];
      console.log(
        `[${new Date().toISOString()}] [INFO] Processing uploaded file: ${
          file.originalname
        }`
      );
      try {
        if (file.mimetype === "application/pdf") {
          const pdfData = await pdf(file.buffer);
          inputText = pdfData.text;
        } else if (file.mimetype === "text/plain") {
          inputText = file.buffer.toString("utf8");
        } else {
          console.log(
            `[${new Date().toISOString()}] [ERROR] Unsupported file type: ${
              file.mimetype
            }`
          );
          return res.status(400).json({
            error: "Unsupported file type. Please upload a PDF or TXT file.",
          });
        }
        console.log(
          `[${new Date().toISOString()}] [INFO] Extracted text from file: ${inputText.slice(
            0,
            100
          )}... (length: ${inputText.length})`
        );
      } catch (error) {
        console.log(
          `[${new Date().toISOString()}] [ERROR] Failed to extract text from file: ${
            error.message
          }`
        );
        return res
          .status(500)
          .json({ error: "Failed to process uploaded file" });
      }
    }

    console.log(
      `[${new Date().toISOString()}] [INFO] Input text: ${inputText.slice(
        0,
        100
      )}... (length: ${inputText.length})`
    );

    if (!inputText.trim()) {
      console.log(
        `[${new Date().toISOString()}] [ERROR] No input text provided`
      );
      return res.status(400).json({
        error: "No input text provided. Please provide text or a valid file.",
      });
    }

    if (!scrapedCorpus.length) {
      console.log(
        `[${new Date().toISOString()}] [ERROR] Corpus not initialized`
      );
      return res
        .status(500)
        .json({ error: "Corpus not initialized, please try again later" });
    }

    console.log(
      `[${new Date().toISOString()}] [INFO] Corpus sample: ${scrapedCorpus[0].slice(
        0,
        100
      )}... (length: ${scrapedCorpus[0].length})`
    );
    console.log(
      `[${new Date().toISOString()}] [INFO] Starting Python script execution for plagiarism detection`
    );

    const startTime = Date.now();
    try {
      const options = {
        mode: "text",
        pythonPath: "/opt/anaconda3/envs/new/bin/python",
        pythonOptions: ["-u"],
        scriptPath: __dirname,
        args: [inputText, JSON.stringify(scrapedCorpus)],
      };

      const shell = new PythonShell("sbert_plagiarism.py", options);

      let stdoutData = "";
      shell.on("message", (message) => {
        stdoutData += message + "\n";
        console.log(
          `[${new Date().toISOString()}] [DEBUG] Python stdout: ${message}`
        );
      });

      shell.on("stderr", (stderr) => {
        console.log(
          `[${new Date().toISOString()}] [DEBUG] Python stderr: ${stderr}`
        );
      });

      shell.on("error", (error) => {
        console.log(
          `[${new Date().toISOString()}] [ERROR] Python script error: ${
            error.message
          }`
        );
      });

      shell.on("close", () => {
        const endTime = Date.now();
        console.log(
          `[${new Date().toISOString()}] [INFO] Python script closed. Time: ${
            endTime - startTime
          }ms`
        );
        try {
          const jsonMatch = stdoutData.match(/\{.*\}/s);
          if (!jsonMatch) {
            throw new Error("No valid JSON found in Python output");
          }
          const result = JSON.parse(jsonMatch[0]);
          console.log(
            `[${new Date().toISOString()}] [INFO] Plagiarism check result: ${JSON.stringify(
              result
            ).slice(0, 100)}...`
          );
          res.json(result);
        } catch (error) {
          console.log(
            `[${new Date().toISOString()}] [ERROR] Failed to parse Python output: ${
              error.message
            }`
          );
          console.log(
            `[${new Date().toISOString()}] [DEBUG] Raw output: ${stdoutData}`
          );
          res.status(500).json({ error: "Failed to process plagiarism check" });
        }
      });
    } catch (error) {
      console.log(
        `[${new Date().toISOString()}] [ERROR] Request failed: ${error.message}`
      );
      res.status(500).json({ error: "Internal server error" });
    }
  }
);

app.listen(port, () => {
  console.log(
    `[${new Date().toISOString()}] [INFO] Backend running on http://localhost:${port}`
  );
});
