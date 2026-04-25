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

const LOGS_DIR = path.join(__dirname, '../../data/logs_db');
if (!fs.existsSync(LOGS_DIR)) fs.mkdirSync(LOGS_DIR, { recursive: true });

const VISITORS_DIR = path.join(__dirname, '../../data/visitors_db');
if (!fs.existsSync(VISITORS_DIR)) fs.mkdirSync(VISITORS_DIR, { recursive: true });

app.use('/blacklist_images', express.static(BLACKLIST_DIR));

const io = new Server(server, { cors: { origin: 'http://localhost:3000', methods: ['GET', 'POST'] } });
const MAX_LOGS = 200;
let securityLogs = [];

app.post('/api/alerts', (req, res) => {
    const newLog = { ...req.body, id: Date.now(), timestamp: new Date().toISOString() };
    securityLogs.unshift(newLog);
    if (securityLogs.length > MAX_LOGS) securityLogs = securityLogs.slice(0, MAX_LOGS);
    io.emit('security_alert', newLog);

    try {
        const dateStr = newLog.timestamp.split('T')[0];
        const logFilePath = path.join(LOGS_DIR, `logs_${dateStr}.json`);
        let dailyLogs = [];
        if (fs.existsSync(logFilePath)) dailyLogs = JSON.parse(fs.readFileSync(logFilePath));
        dailyLogs.unshift(newLog);
        fs.writeFileSync(logFilePath, JSON.stringify(dailyLogs, null, 2));
    } catch(e) { console.error("Could not write daily log", e); }

    return res.status(200).json({ message: 'Alert processed.' });
});

app.get('/api/logs', (req, res) => {
    const { date } = req.query;
    
    // لو المستخدم اختار تاريخ معين من الـ Live Detections
    if (date) {
        const logFilePath = path.join(LOGS_DIR, `logs_${date}.json`);
        if (fs.existsSync(logFilePath)) {
            try { return res.json(JSON.parse(fs.readFileSync(logFilePath))); } catch(e) {}
        }
        return res.json([]);
    }

    // لو مفيش تاريخ (الوضع الافتراضي للـ Dashboard والـ Live)
    // هنقرأ كل الملفات المحفوظة عشان الداتا تفضل موجودة حتى لو عملنا ريستارت
    try {
        let allLogs = [];
        const files = fs.readdirSync(LOGS_DIR).filter(f => f.endsWith('.json'));
        files.sort().reverse(); // عرض الأحدث أولاً
        
        for (const file of files) {
            const data = JSON.parse(fs.readFileSync(path.join(LOGS_DIR, file)));
            allLogs = allLogs.concat(data);
        }
        // نبعت كل الداتا المحفوظة، ولو مفيش نبعت الذاكرة المؤقتة
        return res.json(allLogs.length > 0 ? allLogs : securityLogs);
    } catch(err) {
        return res.json(securityLogs);
    }
});

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

app.get('/api/visitor_image/:name', (req, res) => {
    const name = req.params.name;
    const dirs = fs.readdirSync(VISITORS_DIR);
    for (const dir of dirs) {
        if (fs.statSync(path.join(VISITORS_DIR, dir)).isDirectory()) {
            const p = path.join(VISITORS_DIR, dir, `${name}.jpg`);
            if (fs.existsSync(p)) return res.sendFile(p);
        }
    }
    res.status(404).send('Not found');
});

app.get('/api/visitors', (req, res) => {
    try {
        let visitors = [];
        const dirs = fs.readdirSync(VISITORS_DIR);
        for (const dir of dirs) {
            const dirPath = path.join(VISITORS_DIR, dir);
            if (fs.statSync(dirPath).isDirectory()) {
                const files = fs.readdirSync(dirPath).filter(f => f.match(/\.(jpg|jpeg|png)$/i));
                files.forEach(f => {
                    const name = path.parse(f).name;
                    const stats = fs.statSync(path.join(dirPath, f));
                    visitors.push({
                        id: name,
                        name: name,
                        dateAdded: stats.birthtime.toISOString().split('T')[0],
                        image: `http://localhost:5000/api/visitor_image/${name}?t=${stats.mtimeMs}`,
                        firstSeen: dir
                    });
                });
            }
        }
        res.json(visitors);
    } catch(err) { res.status(500).json({error: err.message}); }
});

app.put('/api/visitors/:oldName', (req, res) => {
    const { newName } = req.body;
    const oldName = req.params.oldName;
    try {
        let renamed = false;
        const dirs = fs.readdirSync(VISITORS_DIR);
        for (const dir of dirs) {
            const dirPath = path.join(VISITORS_DIR, dir);
            if (fs.statSync(dirPath).isDirectory()) {
                const oldPath = path.join(dirPath, `${oldName}.jpg`);
                if (fs.existsSync(oldPath)) {
                    const newPath = path.join(dirPath, `${newName}.jpg`);
                    fs.renameSync(oldPath, newPath);
                    renamed = true;
                }
            }
        }
        if (renamed) {
            // 1. تعديل الاسم في الذاكرة الحية للداشبورد
            securityLogs.forEach(i => { if (i.identity === oldName) i.identity = newName; });
            
            // 2. تعديل الاسم في ملفات السجلات القديمة
            fs.readdirSync(LOGS_DIR).forEach(lf => {
                if (lf.endsWith('.json')) {
                    try {
                        const lfPath = path.join(LOGS_DIR, lf);
                        let data = JSON.parse(fs.readFileSync(lfPath));
                        let changed = false;
                        data.forEach(i => { if (i.identity === oldName) { i.identity = newName; changed = true; } });
                        if (changed) fs.writeFileSync(lfPath, JSON.stringify(data, null, 2));
                    } catch(e) {}
                }
            });

            res.json({ success: true });
            
            // 3. إرسال أمر للكاميرا لتحديث ذاكرتها فوراً
            try { fetch('http://localhost:5001/reload_visitors', { method: 'POST' }); } catch(e) {}
        } else {
            res.status(404).json({ error: 'Visitor not found' });
        }
    } catch(err) { res.status(500).json({error: err.message}); }
});

app.delete('/api/visitors/:name', (req, res) => {
    try {
        const name = req.params.name;
        let deleted = false;
        const dirs = fs.readdirSync(VISITORS_DIR);
        for (const dir of dirs) {
            const dirPath = path.join(VISITORS_DIR, dir);
            if (fs.statSync(dirPath).isDirectory()) {
                const filePath = path.join(dirPath, `${name}.jpg`);
                if (fs.existsSync(filePath)) {
                    fs.unlinkSync(filePath);
                    deleted = true;
                }
            }
        }
        if (deleted) {
            // 1. مسح الزائر من الذاكرة الحية عشان يختفي من الـ Live Detections
            securityLogs = securityLogs.filter(i => i.identity !== name);
            
            // 2. مسح الزائر من ملفات السجلات بالكامل
            fs.readdirSync(LOGS_DIR).forEach(lf => {
                if (lf.endsWith('.json')) {
                    try {
                        const lfPath = path.join(LOGS_DIR, lf);
                        let data = JSON.parse(fs.readFileSync(lfPath));
                        const originalLength = data.length;
                        data = data.filter(i => i.identity !== name);
                        if (data.length !== originalLength) fs.writeFileSync(lfPath, JSON.stringify(data, null, 2));
                    } catch(e) {}
                }
            });

            res.json({ success: true });
            
            // 3. إرسال أمر للكاميرا عشان تبطل تتعرف عليه بالاسم القديم
            try { fetch('http://localhost:5001/reload_visitors', { method: 'POST' }); } catch(e) {}
        }
        else res.status(404).json({ error: 'Visitor not found' });
    } catch(err) { res.status(500).json({error: err.message}); }
});

server.listen(PORT, () => console.log(`Backend running on port ${PORT}`));