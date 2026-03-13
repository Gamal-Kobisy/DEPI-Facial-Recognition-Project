const express  = require('express');
const http     = require('http');
const { Server } = require('socket.io');
const cors     = require('cors');
const fs       = require('fs');
const path     = require('path');

const app    = express();
const server = http.createServer(app);
const PORT   = 5000;


app.use(cors({ origin: 'http://localhost:3000' }));
app.use(express.json({ limit: '50mb' })); 

const BLACKLIST_DIR = path.join(__dirname, '../../data/blacklist_db');
if (!fs.existsSync(BLACKLIST_DIR)) fs.mkdirSync(BLACKLIST_DIR, { recursive: true });

app.use('/blacklist_images', express.static(BLACKLIST_DIR));

const io = new Server(server, { cors: { origin: 'http://localhost:3000', methods: ['GET', 'POST'] } });
const MAX_LOGS = 200;
let securityLogs = [];

app.post('/api/alerts', (req, res) => {
    const newLog = { ...req.body, id: Date.now(), timestamp: new Date().toISOString() };
    securityLogs.unshift(newLog);
    if (securityLogs.length > MAX_LOGS) securityLogs = securityLogs.slice(0, MAX_LOGS);
    io.emit('security_alert', newLog);
    return res.status(200).json({ message: 'Alert processed.' });
});

app.get('/api/logs', (req, res) => res.json(securityLogs));

app.get('/api/blacklist', (req, res) => {
    try {
        const files = fs.readdirSync(BLACKLIST_DIR).filter(f => f.match(/\.(jpg|jpeg|png)$/i));
        const list = files.map(file => {
            const name = path.parse(file).name;
            const stats = fs.statSync(path.join(BLACKLIST_DIR, file));
            return {
                id: name, name: name,
                image: `http://localhost:5000/blacklist_images/${file}?t=${stats.mtimeMs}`,
                dateAdded: stats.mtime.toISOString().split('T')[0]
            };
        });
        res.json(list);
    } catch(err) { res.status(500).json({error: err.message}); }
});

app.post('/api/blacklist', (req, res) => {
    const { oldName, newName, imageBase64 } = req.body;
    try {
        const newPath = path.join(BLACKLIST_DIR, `${newName}.jpg`);
        const oldPath = oldName ? path.join(BLACKLIST_DIR, `${oldName}.jpg`) : null;

        if (imageBase64) {
            const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, "");
            fs.writeFileSync(newPath, base64Data, 'base64');
            if (oldPath && oldName !== newName && fs.existsSync(oldPath)) fs.unlinkSync(oldPath);
        } else if (oldPath && oldName !== newName && fs.existsSync(oldPath)) {
            fs.renameSync(oldPath, newPath);
        }
        res.json({ success: true });
    } catch(err) { res.status(500).json({error: err.message}); }
});


app.delete('/api/blacklist/:name', (req, res) => {
    try {
        const filePath = path.join(BLACKLIST_DIR, `${req.params.name}.jpg`);
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        res.json({ success: true });
    } catch(err) { res.status(500).json({error: err.message}); }
});

server.listen(PORT, () => console.log(`Backend running on port ${PORT}`));