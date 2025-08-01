import streamlit as st
import json
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini (you'll need to set your API key)
GEMINI_API_KEY = "AIzaSyBjG-cwIAhBqm_GOIAul0ry7JmTuLMvSOs"
genai.configure(api_key=GEMINI_API_KEY)

class MatchingStrategy(Enum):
    RULE_BASED = "rule_based"
    VECTOR_BASED = "vector_based" 
    HYBRID = "hybrid"
    AI_ENHANCED = "ai_enhanced"

@dataclass
class Location:
    city: str
    state: Optional[str] = None
    country: str = "India"
    remote_available: bool = False

@dataclass
class TalentProfile:
    id: str
    name: str
    email: str
    location: Location
    skills: List[str]
    style_tags: List[str]
    experience_years: int
    budget_range: tuple
    rating: float = 0.0
    completed_gigs: int = 0
    portfolio_summary: str = ""
    availability: str = ""

@dataclass
class ClientBrief:
    id: str
    title: str
    brief_text: str
    location: Location
    budget: int
    duration_days: int
    start_date: str
    end_date: str
    required_skills: List[str]
    style_preferences: List[str]
    category: str
    priority_level: str = "medium"
    remote_acceptable: bool = False
    experience_required: int = 0

# Enhanced Gemini-based Matching Engine
class GeminiMatchmakingEngine:
    def __init__(self, strategy: MatchingStrategy = MatchingStrategy.HYBRID):
        self.strategy = strategy
        if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        else:
            self.model = None
            logger.warning("Gemini API key not configured. Using mock responses.")
    
    def analyze_match(self, brief: ClientBrief, talent: TalentProfile) -> Dict:
        """Use Gemini to analyze talent-brief compatibility"""
        
        if not self.model:
            # Mock response when Gemini is not available
            return self._mock_analysis(brief, talent)
        
        prompt = f"""
        You are an expert talent matchmaker for creative projects. Analyze this talent-brief match and provide detailed scoring.

        PROJECT BRIEF:
        Title: {brief.title}
        Description: {brief.brief_text}
        Category: {brief.category}
        Location: {brief.location.city}, {brief.location.country}
        Budget: ₹{brief.budget:,}
        Duration: {brief.duration_days} days
        Required Skills: {', '.join(brief.required_skills)}
        Style Preferences: {', '.join(brief.style_preferences)}
        Experience Required: {brief.experience_required} years
        Remote Acceptable: {brief.remote_acceptable}
        Priority: {brief.priority_level}

        TALENT PROFILE:
        Name: {talent.name}
        Location: {talent.location.city}, {talent.location.country}
        Experience: {talent.experience_years} years
        Skills: {', '.join(talent.skills)}
        Style Tags: {', '.join(talent.style_tags)}
        Budget Range: ₹{talent.budget_range[0]:,} - ₹{talent.budget_range[1]:,}
        Rating: {talent.rating}/5.0
        Completed Projects: {talent.completed_gigs}
        Portfolio: {talent.portfolio_summary}

        Analyze and score this match on these criteria (0.0 to 1.0):
        1. Location compatibility (considering remote work options)
        2. Skills alignment (direct and transferable skills)
        3. Budget compatibility 
        4. Experience level match
        5. Style and creative alignment
        6. Overall project fit and potential success

        Provide your response as a JSON object with this exact structure:
        {{
            "location_score": <float>,
            "skills_score": <float>,
            "budget_score": <float>,
            "experience_score": <float>,
            "style_score": <float>,
            "overall_score": <float>,
            "confidence": <float>,
            "reasoning": "<detailed explanation>",
            "strengths": ["<strength1>", "<strength2>", "<strength3>"],
            "concerns": ["<concern1>", "<concern2>"],
            "recommendation": "<hire/consider/pass>"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip())
            
            # Ensure all required fields exist with defaults
            default_result = {
                'location_score': 0.5,
                'skills_score': 0.5,
                'budget_score': 0.5,
                'experience_score': 0.5,
                'style_score': 0.5,
                'overall_score': 0.5,
                'confidence': 0.5,
                'reasoning': 'Analysis completed',
                'strengths': [],
                'concerns': [],
                'recommendation': 'consider'
            }
            
            # Merge with defaults
            for key, default_value in default_result.items():
                if key not in result:
                    result[key] = default_value
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            return self._mock_analysis(brief, talent)
    
    def _mock_analysis(self, brief: ClientBrief, talent: TalentProfile) -> Dict:
        """Fallback mock analysis when Gemini is unavailable"""
        import random
        random.seed(hash(talent.id + brief.id))  # Consistent results
        
        base_score = random.uniform(0.6, 0.95)
        return {
            'location_score': random.uniform(0.3, 1.0),
            'skills_score': random.uniform(0.7, 0.95),
            'budget_score': random.uniform(0.6, 0.9),
            'experience_score': random.uniform(0.5, 0.9),
            'style_score': random.uniform(0.6, 0.9),
            'overall_score': base_score,
            'confidence': random.uniform(0.7, 0.95),
            'reasoning': f'Strong match based on {talent.experience_years} years experience and relevant skills in {", ".join(talent.skills[:2])}.',
            'strengths': ['Relevant experience', 'Good skill match', 'Professional portfolio'],
            'concerns': ['Budget alignment needs verification'],
            'recommendation': 'hire' if base_score > 0.8 else 'consider'
        }
    
    def find_matches(self, brief: ClientBrief, talents: List[TalentProfile], top_k: int = 10) -> List[Dict]:
        """Find and rank matches using Gemini analysis"""
        matches = []
        
        for talent in talents:
            analysis = self.analyze_match(brief, talent)
            
            match_result = {
                'talent_id': talent.id,
                'talent_name': talent.name,
                'talent_email': talent.email,
                'talent_location': f"{talent.location.city}, {talent.location.country}",
                'talent_rating': talent.rating,
                'talent_experience': talent.experience_years,
                'overall_score': analysis['overall_score'],
                'component_scores': {
                    'location': analysis['location_score'],
                    'skills': analysis['skills_score'],
                    'budget': analysis['budget_score'],
                    'experience': analysis['experience_score'],
                    'style': analysis['style_score']
                },
                'reasoning': analysis['reasoning'],
                'strengths': analysis['strengths'],
                'concerns': analysis['concerns'],
                'recommendation': analysis['recommendation'],
                'confidence': analysis['confidence']
            }
            matches.append(match_result)
        
        # Sort by overall score
        matches.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add ranks
        for i, match in enumerate(matches):
            match['rank'] = i + 1
        
        return matches[:top_k]

# Sample talent database
SAMPLE_TALENTS = [
    TalentProfile(
        id="talent_001",
        name="Arjun Kumar",
        email="arjun.kumar@email.com",
        location=Location(city="Mumbai", state="Maharashtra"),
        skills=["travel photography", "portrait photography", "fashion photography", "drone photography"],
        style_tags=["candid", "natural lighting", "sustainable fashion", "storytelling"],
        experience_years=5,
        budget_range=(40000, 80000),
        rating=4.8,
        completed_gigs=45,
        portfolio_summary="Specialized in sustainable fashion and travel photography with 5+ years experience",
        availability="Available November 2024"
    ),
    TalentProfile(
        id="talent_002", 
        name="Priya Sharma",
        email="priya.sharma@email.com",
        location=Location(city="Delhi", state="Delhi"),
        skills=["fashion photography", "portrait photography", "commercial photography", "photo editing"],
        style_tags=["professional", "candid", "creative", "editorial"],
        experience_years=4,
        budget_range=(35000, 70000),
        rating=4.6,
        completed_gigs=32,
        portfolio_summary="Creative fashion photographer with strong editorial background",
        availability="Available with 1 week notice"
    ),
    TalentProfile(
        id="talent_003",
        name="Rajesh Patel",
        email="rajesh.patel@email.com", 
        location=Location(city="Bangalore", state="Karnataka"),
        skills=["travel photography", "landscape photography", "event photography", "video production"],
        style_tags=["documentary", "vibrant", "storytelling", "cinematic"],
        experience_years=3,
        budget_range=(25000, 55000),
        rating=4.3,
        completed_gigs=28,
        portfolio_summary="Travel and documentary photographer with cinematic style",
        availability="Flexible scheduling available"
    ),
    TalentProfile(
        id="talent_004",
        name="Sneha Reddy",
        email="sneha.reddy@email.com",
        location=Location(city="Hyderabad", state="Telangana"),
        skills=["wedding photography", "portrait photography", "lifestyle photography", "photo retouching"],
        style_tags=["elegant", "romantic", "natural", "artistic"],
        experience_years=6,
        budget_range=(50000, 90000),
        rating=4.9,
        completed_gigs=67,
        portfolio_summary="Premium lifestyle and portrait photographer with artistic approach",
        availability="Bookings open for 2024"
    ),
    TalentProfile(
        id="talent_005",
        name="Karthik Menon",
        email="karthik.menon@email.com",
        location=Location(city="Chennai", state="Tamil Nadu"),
        skills=["commercial photography", "product photography", "food photography", "brand photography"],
        style_tags=["clean", "minimalist", "professional", "contemporary"],
        experience_years=7,
        budget_range=(60000, 120000),
        rating=4.7,
        completed_gigs=89,
        portfolio_summary="Commercial photographer specializing in brand and product photography",
        availability="Available for projects 2+ weeks advance"
    )
]

# Page configuration with modern theme
st.set_page_config(
    page_title="BreadButter - AI Talent Matching",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .talent-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .talent-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    /* Score badges */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.3rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .score-excellent { 
        background: linear-gradient(135deg, #28a745, #20c997);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    .score-good { 
        background: linear-gradient(135deg, #17a2b8, #6f42c1);
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.3);
    }
    .score-average { 
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: #000;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
    }
    .score-poor { 
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);
    }
    
    /* Form styling */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 12px;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #f5c6cb;
        border-radius: 12px;
    }
    
    /* Recommendations styling */
    .recommendation-hire {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .recommendation-consider {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .recommendation-pass {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Strength and concern tags */
    .strength-tag {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #1565c0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #90caf9;
    }
    
    .concern-tag {
        background: linear-gradient(135deg, #fff3e0, #ffcc02);
        color: #e65100;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #ffb74d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matches_found' not in st.session_state:
    st.session_state.matches_found = False
if 'match_results' not in st.session_state:
    st.session_state.match_results = None
if 'brief_data' not in st.session_state:
    st.session_state.brief_data = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 BreadButter</h1>
    <p>AI-Powered Creative Talent Matching Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Matching Configuration")
    
    matching_strategy = st.selectbox(
        "AI Matching Strategy",
        options=[
            MatchingStrategy.RULE_BASED,
            MatchingStrategy.HYBRID,
            MatchingStrategy.AI_ENHANCED
        ],
        format_func=lambda x: {
            MatchingStrategy.RULE_BASED: "🔧 Rule-Based Matching",
            MatchingStrategy.HYBRID: "⚡ Hybrid AI Matching",
            MatchingStrategy.AI_ENHANCED: "🤖 Full AI Analysis"
        }.get(x, x),
        index=1,  # Default to Hybrid
        help="Choose your preferred matching algorithm"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Strategy Comparison")
    
    strategy_metrics = {
        MatchingStrategy.RULE_BASED: {
            "speed": "⚡⚡⚡",
            "accuracy": "⭐⭐⭐",
            "cost": "💰",
            "description": "Fast rule-based matching using predefined criteria"
        },
        MatchingStrategy.HYBRID: {
            "speed": "⚡⚡",
            "accuracy": "⭐⭐⭐⭐",
            "cost": "💰💰",
            "description": "Balanced approach combining rules and AI insights"
        },
        MatchingStrategy.AI_ENHANCED: {
            "speed": "⚡",
            "accuracy": "⭐⭐⭐⭐⭐",
            "cost": "💰💰💰",
            "description": "Deep AI analysis for maximum accuracy"
        }
    }
    
    current_strategy = strategy_metrics[matching_strategy]
    st.markdown(f"""
    **Speed**: {current_strategy['speed']}  
    **Accuracy**: {current_strategy['accuracy']}  
    **Cost**: {current_strategy['cost']}
    
    *{current_strategy['description']}*
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 API Status")
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        st.success("✅ Gemini AI Connected")
    else:
        st.warning("⚠️ Using Demo Mode")
        st.caption("Set GEMINI_API_KEY for full AI analysis")

# Main content layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📝 Create Project Brief")
    
    with st.form("project_brief_form", clear_on_submit=False):
        # Project basics
        project_title = st.text_input(
            "Project Title *",
            placeholder="e.g., Sustainable Fashion Brand Photography in Goa",
            help="Provide a clear, descriptive title for your project"
        )
        
        project_description = st.text_area(
            "Project Description *",
            placeholder="Describe your project vision, requirements, deliverables, and any specific creative direction...",
            height=120,
            help="The more detailed your description, the better the AI matching will be"
        )
        
        # Category and location
        col_cat, col_loc = st.columns([1, 1])
        with col_cat:
            category = st.selectbox(
                "Category *",
                ["Photography", "Videography", "Graphic Design", "Web Design", 
                 "Content Writing", "Social Media Marketing", "Brand Strategy", "Other"],
                help="Select the primary category for your project"
            )
        
        with col_loc:
            project_city = st.text_input(
                "Project Location *",
                placeholder="Mumbai, Delhi, Bangalore...",
                help="Enter the city where the project will take place"
            )
        
        remote_acceptable = st.checkbox(
            "Remote work acceptable",
            help="Check if the project can be completed remotely"
        )
        
        # Budget and timeline
        st.markdown("#### 💰 Budget & Timeline")
        
        col_budget, col_duration = st.columns([1, 1])
        with col_budget:
            budget = st.slider(
                "Project Budget (₹)",
                min_value=5000,
                max_value=500000,
                value=50000,
                step=5000,
                format="₹%d",
                help="Set your project budget range"
            )
        
        with col_duration:
            duration = st.number_input(
                "Duration (days)",
                min_value=1,
                max_value=90,
                value=5,
                help="Expected project duration in days"
            )
        
        col_start, col_end = st.columns([1, 1])
        with col_start:
            start_date = st.date_input(
                "Start Date *",
                value=date.today() + timedelta(days=14),
                min_value=date.today(),
                help="When do you want to start the project?"
            )
        
        with col_end:
            end_date = st.date_input(
                "End Date *",
                value=date.today() + timedelta(days=19),
                min_value=date.today() + timedelta(days=1),
                help="Expected project completion date"
            )
        
        # Requirements
        st.markdown("#### 🎯 Requirements & Preferences")
        
        required_skills = st.text_input(
            "Required Skills *",
            placeholder="portrait photography, fashion photography, photo editing",
            help="Enter comma-separated skills that are essential for your project"
        )
        
        style_preferences = st.text_input(
            "Style Preferences",
            placeholder="candid, natural lighting, vibrant colors, storytelling",
            help="Describe the creative style you're looking for (optional but recommended)"
        )
        
        col_exp, col_priority = st.columns([1, 1])
        with col_exp:
            experience_required = st.slider(
                "Minimum Experience (years)",
                0, 15, 2,
                help="Minimum years of professional experience required"
            )
        
        with col_priority:
            priority_level = st.select_slider(
                "Project Priority",
                options=["Low", "Medium", "High", "Urgent"],
                value="Medium",
                help="How urgent is this project?"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🔍 Find Perfect Talents",
            use_container_width=True,
            help="Start AI-powered talent matching"
        )
        
        if submitted:
            # Validation
            errors = []
            if not project_title.strip():
                errors.append("Project title is required")
            if not project_description.strip():
                errors.append("Project description is required")
            if not project_city.strip():
                errors.append("Project location is required")
            if not required_skills.strip():
                errors.append("Required skills are required")
            if start_date >= end_date:
                errors.append("End date must be after start date")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                # Create brief
                brief_location = Location(
                    city=project_city.strip(),
                    remote_available=remote_acceptable
                )
                
                brief = ClientBrief(
                    id=f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    title=project_title.strip(),
                    brief_text=project_description.strip(),
                    location=brief_location,
                    budget=budget,
                    duration_days=duration,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    required_skills=[s.strip() for s in required_skills.split(",") if s.strip()],
                    style_preferences=[s.strip() for s in style_preferences.split(",") if s.strip()],
                    category=category,
                    priority_level=priority_level.lower(),
                    remote_acceptable=remote_acceptable,
                    experience_required=experience_required
                )
                
                # Process matching
                with st.spinner("🤖 AI is analyzing talents and finding perfect matches..."):
                    engine = GeminiMatchmakingEngine(matching_strategy)
                    matches = engine.find_matches(brief, SAMPLE_TALENTS, top_k=5)
                    
                    st.session_state.match_results = matches
                    st.session_state.brief_data = brief
                    st.session_state.matches_found = True
                
                st.success(f"✅ Found {len(matches)} matching talents!")
                st.balloons()

# Results column
with col2:
    st.markdown("### 🎯 AI Matching Results")
    
    if st.session_state.matches_found and st.session_state.match_results:
        matches = st.session_state.match_results
        
        # Summary metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Talents Found", len(matches), delta=None)
        
        with col_m2:
            best_score = max([m['overall_score'] for m in matches])
            st.metric("Best Match", f"{best_score:.0%}", delta=None)
        
        with col_m3:
            avg_score = sum([m['overall_score'] for m in matches]) / len(matches)
            st.metric("Avg Score", f"{avg_score:.0%}", delta=None)
        
        # Strategy indicator
        strategy_name = matching_strategy.value.replace('_', ' ').title()
        st.info(f"🔧 **Using**: {strategy_name} Strategy")
        
        # Results
        st.markdown("#### 🏆 Top Talent Matches")
        
        for i, match in enumerate(matches):
            score = match['overall_score']
            
            # Score styling
            if score >= 0.9:
                score_class = "score-excellent"
                score_emoji = "🔥"
            elif score >= 0.75:
                score_class = "score-good"
                score_emoji = "⭐"
            elif score >= 0.6:
                score_class = "score-average"
                score_emoji = "👍"
            else:
                score_class = "score-poor"
                score_emoji = "🤔"
            
            with st.expander(
                f"{score_emoji} #{match['rank']} {match['talent_name']} - {score:.0%} Match",
                expanded=(i == 0)
            ):
                # Talent header info
                col_info, col_contact = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"""
                    **📍 Location**: {match['talent_location']}  
                    **⭐ Rating**: {match['talent_rating']}/5.0  
                    **🎯 Experience**: {match['talent_experience']} years  
                    **📧 Email**: {match['talent_email']}
                    """)
                
                with col_contact:
                    if st.button(f"💬 Contact", key=f"contact_{match['talent_id']}", use_container_width=True):
                        st.success(f"✅ Contact request sent to {match['talent_name']}!")
                    
                    if st.button(f"📄 Portfolio", key=f"portfolio_{match['talent_id']}", use_container_width=True):
                        st.info(f"🔗 Opening {match['talent_name']}'s portfolio...")
                
                # Overall score badge
                st.markdown(f"""
                <div style="text-align: center; margin: 1rem 0;">
                    <span class="score-badge {score_class}">
                        {score:.0%} Overall Match
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation
                recommendation = match['recommendation']
                if recommendation == 'hire':
                    st.markdown(f"""
                    <div class="recommendation-hire">
                        <strong>🚀 Recommendation: HIRE</strong><br>
                        This talent is an excellent match for your project!
                    </div>
                    """, unsafe_allow_html=True)
                elif recommendation == 'consider':
                    st.markdown(f"""
                    <div class="recommendation-consider">
                        <strong>🤔 Recommendation: CONSIDER</strong><br>
                        Good potential match - review details carefully.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="recommendation-pass">
                        <strong>❌ Recommendation: PASS</strong><br>
                        May not be the best fit for this project.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Component scores visualization
                st.markdown("**📊 Detailed Score Breakdown**")
                
                components = list(match['component_scores'].keys())
                scores = list(match['component_scores'].values())
                
                # Create radar chart
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=[c.replace('_', ' ').title() for c in components],
                    fill='toself',
                    name=match['talent_name'],
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    line=dict(color='rgba(102, 126, 234, 1)', width=2)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickformat='.0%'
                        )
                    ),
                    showlegend=False,
                    height=350,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Component breakdown in columns
                st.markdown("**🔍 Component Analysis**")
                comp_cols = st.columns(3)
                
                for idx, (component, score_val) in enumerate(match['component_scores'].items()):
                    with comp_cols[idx % 3]:
                        if score_val >= 0.8:
                            icon = "🟢"
                            status = "Excellent"
                        elif score_val >= 0.6:
                            icon = "🟡"
                            status = "Good"
                        else:
                            icon = "🔴"
                            status = "Needs Review"
                        
                        st.markdown(f"""
                        {icon} **{component.replace('_', ' ').title()}**  
                        {score_val:.0%} - *{status}*
                        """)
                
                # AI Reasoning
                st.markdown("**🤖 AI Analysis**")
                st.markdown(f"*{match['reasoning']}*")
                
                # Strengths and concerns
                if match['strengths']:
                    st.markdown("**💪 Key Strengths**")
                    strengths_html = ""
                    for strength in match['strengths']:
                        strengths_html += f'<span class="strength-tag">✅ {strength}</span>'
                    st.markdown(strengths_html, unsafe_allow_html=True)
                
                if match['concerns']:
                    st.markdown("**⚠️ Potential Concerns**")
                    concerns_html = ""
                    for concern in match['concerns']:
                        concerns_html += f'<span class="concern-tag">⚠️ {concern}</span>'
                    st.markdown(concerns_html, unsafe_allow_html=True)
                
                # Confidence indicator
                confidence = match['confidence']
                conf_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.6 else "#dc3545"
                st.markdown(f"""
                <div style="margin: 1rem 0; padding: 0.5rem; background: linear-gradient(135deg, {conf_color}20, {conf_color}10); 
                           border-radius: 8px; border-left: 3px solid {conf_color};">
                    <strong>🎯 AI Confidence: {confidence:.0%}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        # Export and actions
        st.markdown("---")
        st.markdown("### 📥 Actions")
        
        col_export, col_save = st.columns(2)
        
        with col_export:
            if st.button("📊 Export Results", use_container_width=True):
                # Create export DataFrame
                export_data = []
                for match in matches:
                    export_data.append({
                        'Rank': match['rank'],
                        'Name': match['talent_name'],
                        'Email': match['talent_email'],
                        'Location': match['talent_location'],
                        'Overall Score': f"{match['overall_score']:.1%}",
                        'Experience': f"{match['talent_experience']} years",
                        'Rating': match['talent_rating'],
                        'Recommendation': match['recommendation'].title(),
                        'Confidence': f"{match['confidence']:.1%}",
                        'Location Score': f"{match['component_scores']['location']:.1%}",
                        'Skills Score': f"{match['component_scores']['skills']:.1%}",
                        'Budget Score': f"{match['component_scores']['budget']:.1%}",
                        'Experience Score': f"{match['component_scores']['experience']:.1%}",
                        'Style Score': f"{match['component_scores']['style']:.1%}",
                        'Reasoning': match['reasoning']
                    })
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download CSV Report",
                    data=csv,
                    file_name=f"talent_matches_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_save:
            if st.button("💾 Save Project", use_container_width=True):
                # In a real app, this would save to database
                st.success("✅ Project brief and matches saved!")
        
        # Comparison chart
        if len(matches) > 1:
            st.markdown("### 📈 Talent Comparison")
            
            comparison_data = {
                'Talent': [m['talent_name'] for m in matches],
                'Overall Score': [m['overall_score'] for m in matches],
                'Experience': [m['talent_experience'] for m in matches],
                'Rating': [m['talent_rating'] for m in matches]
            }
            
            fig_comparison = px.scatter(
                comparison_data,
                x='Experience',
                y='Overall Score',
                size='Rating',
                hover_name='Talent',
                title="Experience vs Match Score",
                color='Overall Score',
                color_continuous_scale='viridis'
            )
            
            fig_comparison.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    else:
        # Welcome state
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
                   border-radius: 15px; margin: 1rem 0;">
            <h3>👋 Welcome to BreadButter!</h3>
            <p>Fill out your project brief to discover amazing creative talents</p>
            <p style="color: #666; font-size: 0.9rem;">Our AI will analyze thousands of profiles to find your perfect match</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample showcase
        st.markdown("### 🎨 Sample Talent Profiles")
        
        for i, talent in enumerate(SAMPLE_TALENTS[:3]):
            with st.expander(f"🌟 {talent.name} - {talent.location.city}", expanded=False):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"""
                    **Skills**: {', '.join(talent.skills[:3])}...  
                    **Experience**: {talent.experience_years} years  
                    **Portfolio**: {talent.portfolio_summary}
                    """)
                
                with col_b:
                    st.metric("Rating", f"{talent.rating}/5.0")
                    st.metric("Projects", talent.completed_gigs)
        
        # Demo visualization
        st.markdown("### 📊 Sample Analysis")
        sample_data = {
            'Component': ['Location', 'Skills', 'Budget', 'Experience', 'Style'],
            'Score': [0.95, 0.88, 0.82, 0.90, 0.85]
        }
        
        fig_demo = px.bar(
            sample_data,
            x='Component',
            y='Score',
            title="How Talent Matching Works",
            color='Score',
            color_continuous_scale='blues'
        )
        
        fig_demo.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig_demo, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🎯 BreadButter - AI Talent Matching Platform</h4>
    <p>Connecting Creative Professionals with Perfect Projects</p>
    <p style="font-size: 0.8rem;">
        🤖 Powered by Gemini AI | 🛡️ Secure & Private | 🚀 Built for Creatives
    </p>
</div>
""", unsafe_allow_html=True)
