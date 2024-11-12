from flask import Flask, render_template, request, jsonify
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import markdown2
from werkzeug.utils import secure_filename
import PyPDF2
import io
import re

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
PROJECT_ID = "winter-cogency-436501-g9"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(pdf_file):
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pdf_text

def call_gemini_flash(prompt, temperature=1.0):  # Set default temperature to 1.0
    """Call Gemini Flash model with given prompt"""
    model = GenerativeModel("gemini-1.5-flash-002")
    response = model.generate_content(
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 8000,
            "top_p": 1.0,
            "top_k": 40
        }
    )
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            pdf_text = extract_pdf_text(file)
            return jsonify({'text': pdf_text})
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.json
        topic = data['topic']
        fields = data['fields']
        
        prompt = f"""Based on this information about a {topic} problem:
        
        Goal: {fields['goal']}
        Key Questions: {fields['keyQuestions']}
        Stakeholders: {fields['stakeholders']}
        Constraints: {fields['constraints']}
        Priorities: {fields['priorities']}
        Resources: {fields['resources']}
        Additional Info: {fields['additionalInfo']}
        
        Generate 5-10 focused follow-up questions to better understand this situation.
        Return only the numbered questions, one per line.
        """
        
        response = call_gemini_flash(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip() and any(c.isdigit() for c in q)]
        
        return jsonify({'questions': questions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-title', methods=['POST'])
def generate_title():
    try:
        data = request.json
        outline = data['outline']
        context = data['context']
        
        title_prompt = f"""Analyze this outline and context for a strategic solution:
        {outline}
        {context}
        
        Create an evocative, imaginative title that captures the essence of this analysis.
        The title should:
        - Use creative metaphors, imagery, or powerful concepts
        - Be memorable and thought-provoking
        - Reflect the transformative nature of the solution
        - Avoid generic words like 'report', 'analysis', or 'solution'
        - Be bold and innovative
        - Create an immediate emotional or intellectual impact
        
        Examples of the style (but don't use these):
        - "Quantum Leap: Orchestrating Tomorrow's Symphony"
        - "Phoenix Rising: The Digital Renaissance Blueprint"
        - "Constellation of Change: Mapping New Horizons"
        """
        
        title = call_gemini_flash(title_prompt)
        return jsonify({'title': title.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/generate-outline', methods=['POST'])
def generate_outline():
    try:
        data = request.json
        context = f"""Topic: {data['topic']}
        
        Initial Information:
        Goal: {data['fields']['goal']}
        Key Questions: {data['fields']['keyQuestions']}
        Stakeholders: {data['fields']['stakeholders']}
        Constraints: {data['fields']['constraints']}
        Priorities: {data['fields']['priorities']}
        Resources: {data['fields']['resources']}
        Additional Info: {data['fields']['additionalInfo']}
        
        Follow-up Questions and Answers:
        {' '.join([f'Q{i+1}: {q}\nA{i+1}: {a}' for i, (q, a) in enumerate(zip(data['questions'], data['answers']))])}
        """
        
        outline_prompt = f"""Based on this context:
        {context}
        
        Create a comprehensive outline with exactly 10 chapters.
        Each chapter should explore a crucial aspect of the solution.
        Format in markdown using '## Chapter X:' for each chapter title.
        Make the chapter titles creative and compelling.
        """
        
        outline = call_gemini_flash(outline_prompt)
        
        return jsonify({
            'outline': markdown2.markdown(outline),
            'markdown': outline,
            'context': context
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-full-solution', methods=['POST'])
def generate_full_solution():
    try:
        data = request.json
        context = data['context']
        outline = data['outline']
        feedback = data['feedback']
        pdf_content = data.get('pdf_content', '')
        title = data.get('title', '')
        
        if pdf_content:
            context += f"\n\nAdditional Reference Material:\n{pdf_content}"

        # Generate introduction
        intro_prompt = f"""Based on this context and outline:
        {context}
        {outline}
        
        Generate a comprehensive introduction for this analysis.
        Create an engaging, insightful opening that:
        - Frames the core challenges and opportunities
        - Sets up the key themes and ideas
        - Captures the reader's interest
        - Establishes the significance of this analysis
        
        Write in a bold, compelling style.
        """
        
        introduction = call_gemini_flash(intro_prompt)
        
        # Extract chapter titles
        chapter_titles = re.findall(r'## Chapter \d+:(.*?)\n', outline)
        
        # Ensure we have exactly 10 chapters
        if len(chapter_titles) != 10:
            outline_prompt = f"""Based on this context:
            {context}
            
            Create exactly 10 chapter titles for this analysis.
            Format each as '## Chapter X: Title'
            Make each title creative and compelling.
            """
            chapter_outline = call_gemini_flash(outline_prompt)
            chapter_titles = re.findall(r'## Chapter \d+:(.*?)\n', chapter_outline)
        
        # Generate chapters
        chapters = []
        chapter_context = introduction
        
        for i, title in enumerate(chapter_titles, 1):
            chapter_prompt = f"""Based on this context:
            {context}
            
            Previous content generated:
            {chapter_context}
            
            Write Chapter {i}: {title}
            
            Create an engaging, insightful chapter that:
            - Deeply explores this aspect of the analysis
            - Provides unique insights and perspectives
            - Connects to the broader themes
            - Offers practical, actionable ideas
            
            Write in a bold, compelling style.
            """
            
            chapter_content = call_gemini_flash(chapter_prompt)
            chapters.append(f"## Chapter {i}: {title.strip()}\n\n{chapter_content}")
            chapter_context += chapter_content
        
        # Generate conclusion
        conclusion_prompt = f"""Based on all previous content:
        {chapter_context}
        
        Write a powerful conclusion that:
        - Synthesizes the key insights
        - Offers bold, actionable recommendations
        - Paints a compelling vision of the future
        - Leaves the reader inspired
        
        Write in a bold, compelling style.
        """
        
        conclusion = call_gemini_flash(conclusion_prompt)
        
        # Combine all sections with prominent title display
        full_solution = f"""# {title}

## Introduction
{introduction}

{''.join(chapters)}

## Conclusion
{conclusion}"""
        
        return jsonify({
            'solution': markdown2.markdown(full_solution),
            'markdown': full_solution
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)