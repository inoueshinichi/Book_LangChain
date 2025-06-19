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
        access_token: process.env.OPENAI_API_TOKEN,
    })
})

app.post('/', async (req, res, next) => {
    prompt = req.body.prompt
    console.log('prompt:', prompt)
    // const result = await access_openai_1(prompt)
    const result = await access_openai_2(prompt)
    if (result === null) {
        res.status(500)
    }
    res.render('index', {
        question: prompt, 
        result: result, 
        access_token: process.env.OPENAI_API_TOKEN,
    })
})

app.get('/openai/models', async (req, res, next) => {
    const openai = new OpenAI({
        apiKey: process.env.OPENAI_API_TOKEN, // 環境変数からAPIキーを読み込み
    })

    let retText
    try {
        console.log('OpenAIのAPIで利用できるモデルの一覧を取得中')
        const response = await openai.models.list()
        console.log('\n--- OpenAI 利用可能なモデル一覧 ---')
        response.data.forEach(model => {
            // console.log(`ID: ${model.id}`);
            // console.log(`  Created: ${new Date(model.created * 1000).toLocaleString()}`); // タイムスタンプを日付に変換
            // console.log(`  Owned By: ${model.owned_by}`);
            // console.log('------------------------------------');
            retText += 
            `
            ID: ${model.id}
                Created: ${new Date(model.created * 1000).toLocaleString()}
                Owned By: ${model.owned_by}
            ------------------------------------
            `
        })

        console.log(retText)
        
    } catch(error) {
        if (error instanceof OpenAI.APIError) {
            console.error('APIError', error)
        } else {
            console.error('Unkonwn error for OpenAI API')
        }

        retText = null
    }

    res.json({
        "models": retText
    })
})

async function access_openai_1(prompt_value) { 
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


async function access_openai_2(prompt_value) {
    let result
    try {
        const response = await client.completions.create({
            model: "gpt-3.5-turbo-instruct",
            prompt: prompt_value,
            // stop: "。",
            max_tokens: 300,
            temperature: 0.7
        })
        result =  response.choices[0].text.trim()
        console.log(result)
    } catch(error) {
        result = null
        if (error instanceof OpenAI.APIConnectionTimeoutError) {
            console.error('APIConnectionTimeoutError', error)
        }
        else if (error instanceof OpenAI.AuthenticationError) {
            console.error('AuthenticationError', error)
        } 
        else if (error instanceof OpenAI.APIConnectionError) {
            console.error('APIConnectionError', error)
        }
        else if (error instanceof OpenAI.InternalServerError) {
            console.error('InternalServerError', error)
        }
        else if (error instanceof OpenAI.RateLimitError) {
            console.error('RateLimitError', error)
        }
        else if (error instanceof OpenAI.BadRequestError) {
            console.log('BadRequestError', error)
        }
        else if (error instanceof OpenAI.NotFoundError) {
            console.log('NotFoundError', error)
        }
        else if (error instanceof OpenAI.APIError) {
            console.error('APIError', error)
        } 
    }
    return result
}

app.listen(port=8002, () => {
    console.log(`Start express server http://localhost:8002/`)
})