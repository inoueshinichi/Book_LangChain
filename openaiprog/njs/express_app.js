require('dotenv').config()
const express = require('express')
app = express()

// EJSをテンプレートエンジンとして設定
app.set('view engine', 'ejs')

// テンプレートファイルが置かれているディレクトリを指定
// ここでは、プロジェクトルートの 'views' ディレクトリを指定
app.set('views', './views')

// 静的ファイル（CSS、JavaScript、画像など）を配信するための設定
// 'public' ディレクトリ内のファイルを '/static' パスでアクセス可能にする
app.use('/static', express.static('public'))

// --- ★ ここが重要 ★ ---
// URLエンコードされたボディを解析するためのミドルウェア
// HTMLフォームから送られる標準的なデータ形式 (application/x-www-form-urlencoded) を処理
app.use(express.urlencoded({ extended: true }))
// JSON形式のボディを解析するためのミドルウェア (APIなどでJSONデータを受け取る場合)
// app.use(express.json());
// -----------------------

const OpenAI = require('openai')

const client = new OpenAI({
    apiKey: process.env.OPENAI_API_TOKEN, // 環境変数からAPIキーを読み込み
  });

app.get('/', (req, res, next) => {
    res.render('index', {
        question: null,
        result: null,
    })
})

app.post('/', async (req, res, next) => {
    prompt = req.body.prompt
    console.log('prompt:', prompt)
    const result = await access_openai(prompt)
    if (result === null) {
        res.status(500)
    }
    res.render('index', {
        question: prompt, result: result
    })
})

async function access_openai(prompt_value) { 
    let result
    try {
        const response = await client.completions.create({
            model: "gpt-3.5-turbo-instruct", 
            prompt: prompt_value,
            stop: "。",
            max_tokens: 100,
            temperature: 0.5
        })
        result =  response.choices[0].text.trim()
        console.log(result)
    } catch (error) {
        console.error('Error calling OpenAI API:', error)
        result = null
    }
    return result
}

app.listen(port=8002, () => {
    console.log(`Start express server http://localhost:8002/`)
})