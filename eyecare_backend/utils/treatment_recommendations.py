"""
Enhanced Treatment Recommendation System for EyeCare AI
Provides comprehensive, evidence-based treatment recommendations for eye diseases
"""
from typing import List, Dict, Tuple
from enum import Enum
from datetime import datetime, timedelta


class SeverityLevel(Enum):
    """Severity levels for eye conditions"""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class UrgencyLevel(Enum):
    """Urgency levels for medical attention"""
    ROUTINE = "routine"           # Regular check-up schedule
    MONITOR = "monitor"           # Monitor and schedule soon
    PROMPT = "prompt"             # See doctor within days
    URGENT = "urgent"             # See doctor within hours
    EMERGENCY = "emergency"       # Immediate medical attention


class TreatmentRecommendationSystem:
    """
    Comprehensive treatment recommendation system for eye diseases
    """
    
    def __init__(self):
        self.recommendations_db = self._initialize_recommendations()
        
    def _initialize_recommendations(self) -> Dict:
        """Initialize the comprehensive recommendations database"""
        return {
            "normal": {
                "severity_levels": {
                    SeverityLevel.NORMAL: {
                        "urgency": UrgencyLevel.ROUTINE,
                        "immediate_actions": [
                            "✅ Your eye examination appears normal",
                            "Continue current eye care routine"
                        ],
                        "treatment_options": [
                            "No specific treatment required",
                            "Maintain preventive eye care"
                        ],
                        "lifestyle_recommendations": [
                            "Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds",
                            "Wear UV-protective sunglasses when outdoors",
                            "Maintain a diet rich in omega-3 fatty acids, vitamins A, C, and E",
                            "Stay hydrated and get adequate sleep",
                            "Avoid smoking and limit alcohol consumption"
                        ],
                        "follow_up": [
                            "Schedule routine eye exam every 1-2 years (or as advised by your eye care professional)",
                            "Monitor for any changes in vision",
                            "Contact eye care provider if symptoms develop"
                        ],
                        "emergency_signs": [
                            "Sudden vision loss",
                            "Severe eye pain",
                            "Flashing lights or new floaters",
                            "Curtain-like shadow across vision"
                        ]
                    }
                }
            },
            
            "diabetic_retinopathy": {
                "severity_levels": {
                    SeverityLevel.MILD: {
                        "urgency": UrgencyLevel.PROMPT,
                        "immediate_actions": [
                            "🔍 Mild diabetic retinopathy detected",
                            "Schedule ophthalmologist consultation within 1-2 weeks",
                            "Begin or intensify blood glucose monitoring"
                        ],
                        "treatment_options": [
                            "Strict diabetes management is the primary treatment",
                            "Regular dilated eye exams every 6-12 months",
                            "Blood pressure and cholesterol control",
                            "Consider diabetes education classes"
                        ],
                        "lifestyle_recommendations": [
                            "Maintain blood glucose levels as close to normal as possible",
                            "Follow diabetic diet plan consistently",
                            "Exercise regularly as approved by your doctor",
                            "Take prescribed medications as directed",
                            "Monitor blood pressure daily",
                            "Quit smoking immediately if you smoke"
                        ],
                        "follow_up": [
                            "Ophthalmologist every 6-12 months",
                            "Endocrinologist every 3-4 months",
                            "Primary care physician monthly until stable"
                        ]
                    },
                    SeverityLevel.MODERATE: {
                        "urgency": UrgencyLevel.URGENT,
                        "immediate_actions": [
                            "⚠️ Moderate diabetic retinopathy detected - URGENT",
                            "Contact ophthalmologist immediately",
                            "Same-day or next-day appointment required"
                        ],
                        "treatment_options": [
                            "Anti-VEGF injections may be recommended",
                            "Laser photocoagulation therapy possible",
                            "Intensive diabetes management required",
                            "Blood pressure optimization essential"
                        ],
                        "lifestyle_recommendations": [
                            "Extremely strict blood sugar control (HbA1c < 7%)",
                            "Daily blood glucose monitoring",
                            "Immediate smoking cessation",
                            "Limit sodium intake for blood pressure control",
                            "Regular cardiovascular exercise as tolerated"
                        ],
                        "follow_up": [
                            "Ophthalmologist every 3-4 months",
                            "Retinal specialist consultation",
                            "Endocrinologist monthly"
                        ]
                    },
                    SeverityLevel.SEVERE: {
                        "urgency": UrgencyLevel.EMERGENCY,
                        "immediate_actions": [
                            "🚨 SEVERE diabetic retinopathy - EMERGENCY",
                            "Seek immediate medical attention",
                            "Call ophthalmologist or go to emergency room"
                        ],
                        "treatment_options": [
                            "Immediate anti-VEGF therapy likely required",
                            "Panretinal photocoagulation may be necessary",
                            "Possible vitrectomy surgery",
                            "Intensive care coordination required"
                        ],
                        "lifestyle_recommendations": [
                            "Hospital-grade diabetes management",
                            "Complete lifestyle modification under medical supervision",
                            "Immediate cessation of all risk factors"
                        ],
                        "follow_up": [
                            "Weekly to monthly ophthalmologist visits",
                            "Immediate retinal specialist care",
                            "Multidisciplinary diabetes team involvement"
                        ]
                    }
                }
            },
            
            "glaucoma": {
                "severity_levels": {
                    SeverityLevel.MILD: {
                        "urgency": UrgencyLevel.PROMPT,
                        "immediate_actions": [
                            "🔍 Possible glaucoma detected",
                            "Schedule comprehensive eye examination within 1 week",
                            "Avoid activities that increase intraocular pressure"
                        ],
                        "treatment_options": [
                            "Prescription eye drops to lower eye pressure",
                            "Regular intraocular pressure monitoring",
                            "Visual field testing every 6 months",
                            "Optic nerve imaging (OCT)"
                        ],
                        "lifestyle_recommendations": [
                            "Avoid inverted positions (head below heart) for extended periods",
                            "Limit caffeine intake",
                            "Stay hydrated but avoid drinking large amounts quickly",
                            "Regular moderate exercise (avoid heavy lifting)",
                            "Protect eyes from trauma"
                        ],
                        "follow_up": [
                            "Ophthalmologist every 3-6 months",
                            "Regular pressure checks",
                            "Annual visual field tests"
                        ]
                    },
                    SeverityLevel.MODERATE: {
                        "urgency": UrgencyLevel.URGENT,
                        "immediate_actions": [
                            "⚠️ Glaucoma progression detected - URGENT",
                            "Schedule same-day ophthalmologist consultation",
                            "Begin pressure-lowering measures immediately"
                        ],
                        "treatment_options": [
                            "Multiple eye drops may be prescribed",
                            "Laser therapy (trabeculoplasty) consideration",
                            "Possible surgical intervention discussion",
                            "Combination therapy protocols"
                        ],
                        "lifestyle_recommendations": [
                            "Strict adherence to medication schedule",
                            "Complete avoidance of high-pressure activities",
                            "Stress reduction techniques",
                            "Regular sleep schedule"
                        ],
                        "follow_up": [
                            "Ophthalmologist every 2-3 months",
                            "Glaucoma specialist consultation",
                            "Monthly pressure monitoring"
                        ]
                    },
                    SeverityLevel.SEVERE: {
                        "urgency": UrgencyLevel.EMERGENCY,
                        "immediate_actions": [
                            "🚨 SEVERE glaucoma or acute attack - EMERGENCY",
                            "Go to emergency room immediately",
                            "Do not delay - vision loss may be permanent"
                        ],
                        "treatment_options": [
                            "Emergency pressure-lowering treatment",
                            "Immediate surgical intervention likely",
                            "Trabeculectomy or tube shunt surgery",
                            "Intensive medical therapy"
                        ],
                        "lifestyle_recommendations": [
                            "Complete activity restriction until cleared",
                            "Immediate medical supervision required"
                        ],
                        "follow_up": [
                            "Daily to weekly monitoring initially",
                            "Surgical follow-up protocol",
                            "Lifelong specialist care"
                        ]
                    }
                }
            },
            
            "cataract": {
                "severity_levels": {
                    SeverityLevel.MILD: {
                        "urgency": UrgencyLevel.MONITOR,
                        "immediate_actions": [
                            "👁️ Early cataract detected",
                            "Schedule routine ophthalmologist consultation",
                            "Monitor for vision changes"
                        ],
                        "treatment_options": [
                            "No immediate treatment required",
                            "Update eyeglass prescription as needed",
                            "Monitor progression with regular exams",
                            "Consider surgery when vision affects daily life"
                        ],
                        "lifestyle_recommendations": [
                            "Use brighter lighting for reading and close work",
                            "Wear anti-glare sunglasses outdoors",
                            "Use magnifying glasses when needed",
                            "Avoid driving at night if vision is impaired",
                            "Maintain good nutrition with antioxidants"
                        ],
                        "follow_up": [
                            "Annual eye examinations",
                            "Vision monitoring every 6 months",
                            "Contact doctor if vision worsens"
                        ]
                    },
                    SeverityLevel.MODERATE: {
                        "urgency": UrgencyLevel.PROMPT,
                        "immediate_actions": [
                            "⚠️ Moderate cataract affecting vision",
                            "Schedule ophthalmologist consultation within 2 weeks",
                            "Consider cataract surgery evaluation"
                        ],
                        "treatment_options": [
                            "Cataract surgery consultation",
                            "Pre-operative evaluation",
                            "Intraocular lens selection discussion",
                            "Temporary vision aids while awaiting surgery"
                        ],
                        "lifestyle_recommendations": [
                            "Use maximum lighting for all activities",
                            "Avoid driving in challenging conditions",
                            "Use contrast-enhancing tools",
                            "Prepare home environment for safety"
                        ],
                        "follow_up": [
                            "Surgeon consultation for timing",
                            "Pre-operative testing",
                            "Regular monitoring until surgery"
                        ]
                    },
                    SeverityLevel.SEVERE: {
                        "urgency": UrgencyLevel.URGENT,
                        "immediate_actions": [
                            "🚨 Severe cataract - vision significantly impaired",
                            "Schedule urgent cataract surgery consultation",
                            "Cease driving and hazardous activities"
                        ],
                        "treatment_options": [
                            "Immediate cataract surgery recommendation",
                            "Emergency surgery if complications present",
                            "Premium intraocular lens options",
                            "Bilateral surgery planning if needed"
                        ],
                        "lifestyle_recommendations": [
                            "Complete activity modification for safety",
                            "Arrange assistance for daily activities",
                            "Home safety modifications"
                        ],
                        "follow_up": [
                            "Immediate surgical scheduling",
                            "Pre-operative optimization",
                            "Post-operative care planning"
                        ]
                    }
                }
            },
            
            # Additional conditions for future expansion
            "age_related_macular_degeneration": {
                "severity_levels": {
                    SeverityLevel.MILD: {
                        "urgency": UrgencyLevel.PROMPT,
                        "immediate_actions": [
                            "🔍 Early AMD signs detected",
                            "Schedule retinal specialist consultation within 2 weeks",
                            "Begin Amsler grid monitoring at home"
                        ],
                        "treatment_options": [
                            "High-dose antioxidant vitamins (AREDS2 formula)",
                            "Dietary modification with leafy greens",
                            "Regular monitoring with OCT imaging",
                            "Lifestyle risk factor modification"
                        ],
                        "lifestyle_recommendations": [
                            "Stop smoking immediately",
                            "Eat diet rich in lutein and zeaxanthin",
                            "Protect eyes from UV light",
                            "Maintain healthy weight and blood pressure",
                            "Use Amsler grid daily for monitoring"
                        ],
                        "follow_up": [
                            "Retinal specialist every 6 months",
                            "Daily Amsler grid self-monitoring",
                            "Annual comprehensive eye examination"
                        ]
                    }
                }
            },
            
            "hypertensive_retinopathy": {
                "severity_levels": {
                    SeverityLevel.MILD: {
                        "urgency": UrgencyLevel.PROMPT,
                        "immediate_actions": [
                            "🔍 Blood pressure changes affecting retina",
                            "Schedule both eye doctor and primary care visits",
                            "Begin blood pressure monitoring"
                        ],
                        "treatment_options": [
                            "Blood pressure management is primary treatment",
                            "Antihypertensive medication optimization",
                            "Regular retinal monitoring",
                            "Cardiovascular risk assessment"
                        ],
                        "lifestyle_recommendations": [
                            "Reduce sodium intake significantly",
                            "Increase physical activity gradually",
                            "Weight reduction if overweight",
                            "Stress management techniques",
                            "Regular blood pressure monitoring"
                        ],
                        "follow_up": [
                            "Primary care physician within 1 week",
                            "Ophthalmologist every 3-6 months",
                            "Daily blood pressure monitoring"
                        ]
                    }
                }
            }
        }
    
    def get_comprehensive_recommendations(
        self, 
        predicted_class: str, 
        confidence: float,
        severity_indicators: Dict = None
    ) -> Dict:
        """
        Get comprehensive treatment recommendations based on prediction and severity
        
        Args:
            predicted_class: The predicted eye condition
            confidence: Prediction confidence (0-100)
            severity_indicators: Additional indicators for severity assessment
        
        Returns:
            Comprehensive recommendation dictionary
        """
        # Determine severity level based on confidence and additional indicators
        severity = self._assess_severity(predicted_class, confidence, severity_indicators)
        
        # Get base recommendations
        condition_data = self.recommendations_db.get(predicted_class.lower(), {})
        severity_data = condition_data.get("severity_levels", {}).get(severity, {})
        
        if not severity_data:
            # Fallback for unrecognized conditions
            return self._get_fallback_recommendations(predicted_class, confidence)
        
        # Build comprehensive response
        recommendations = {
            "condition": predicted_class,
            "confidence": confidence,
            "severity_level": severity.value,
            "urgency_level": severity_data.get("urgency", UrgencyLevel.ROUTINE).value,
            "timestamp": datetime.utcnow().isoformat(),
            
            "immediate_actions": severity_data.get("immediate_actions", []),
            "treatment_options": severity_data.get("treatment_options", []),
            "lifestyle_recommendations": severity_data.get("lifestyle_recommendations", []),
            "follow_up_care": severity_data.get("follow_up", []),
            
            "emergency_warning": self._get_emergency_warning(severity_data.get("urgency")),
            "next_appointment": self._calculate_next_appointment(severity_data.get("urgency")),
            
            # Additional context
            "educational_resources": self._get_educational_resources(predicted_class),
            "support_contacts": self._get_support_contacts(),
        }
        
        # Add confidence-specific notes
        recommendations["confidence_notes"] = self._get_confidence_notes(confidence, severity)
        
        return recommendations
    
    def _assess_severity(self, condition: str, confidence: float, indicators: Dict = None) -> SeverityLevel:
        """Assess severity level based on multiple factors"""
        
        if condition.lower() == "normal":
            return SeverityLevel.NORMAL
        
        # Base severity on confidence level
        if confidence >= 90:
            base_severity = SeverityLevel.SEVERE
        elif confidence >= 75:
            base_severity = SeverityLevel.MODERATE
        elif confidence >= 60:
            base_severity = SeverityLevel.MILD
        else:
            base_severity = SeverityLevel.MILD
        
        # Adjust based on additional indicators (future enhancement)
        if indicators:
            # This could include factors like:
            # - Patient age, medical history
            # - Multiple conditions detected
            # - Image quality indicators
            # - Previous examination results
            pass
        
        return base_severity
    
    def _get_confidence_notes(self, confidence: float, severity: SeverityLevel) -> List[str]:
        """Generate confidence-specific notes"""
        notes = []
        
        if confidence >= 90:
            notes.append(f"High confidence detection ({confidence:.1f}%) - recommendations are strongly indicated")
        elif confidence >= 75:
            notes.append(f"Good confidence detection ({confidence:.1f}%) - follow recommendations as suggested")
        elif confidence >= 60:
            notes.append(f"Moderate confidence detection ({confidence:.1f}%) - consider professional confirmation")
        else:
            notes.append(f"Lower confidence detection ({confidence:.1f}%) - professional examination strongly recommended")
            notes.append("This AI screening should not replace professional medical examination")
        
        return notes
    
    def _get_emergency_warning(self, urgency: UrgencyLevel) -> Dict:
        """Get emergency warning based on urgency level"""
        warnings = {
            UrgencyLevel.EMERGENCY: {
                "level": "CRITICAL",
                "message": "🚨 MEDICAL EMERGENCY: Seek immediate medical attention. Do not delay.",
                "action": "Go to emergency room or call emergency services",
                "timeframe": "Immediately"
            },
            UrgencyLevel.URGENT: {
                "level": "HIGH",
                "message": "⚠️ URGENT: Medical attention required within hours to days",
                "action": "Contact ophthalmologist immediately for same-day appointment",
                "timeframe": "Within 24 hours"
            },
            UrgencyLevel.PROMPT: {
                "level": "MODERATE",
                "message": "📞 Schedule medical consultation promptly",
                "action": "Contact eye care professional within 1-2 weeks",
                "timeframe": "Within 1-2 weeks"
            },
            UrgencyLevel.MONITOR: {
                "level": "LOW",
                "message": "👁️ Monitor condition and schedule routine care",
                "action": "Schedule routine eye examination",
                "timeframe": "Within 1-3 months"
            },
            UrgencyLevel.ROUTINE: {
                "level": "MINIMAL",
                "message": "✅ Continue routine eye care",
                "action": "Follow regular eye care schedule",
                "timeframe": "As per normal schedule"
            }
        }
        
        return warnings.get(urgency, warnings[UrgencyLevel.ROUTINE])
    
    def _calculate_next_appointment(self, urgency: UrgencyLevel) -> Dict:
        """Calculate recommended next appointment timing"""
        now = datetime.utcnow()
        
        timing_map = {
            UrgencyLevel.EMERGENCY: timedelta(hours=0),  # Immediate
            UrgencyLevel.URGENT: timedelta(days=1),
            UrgencyLevel.PROMPT: timedelta(weeks=1),
            UrgencyLevel.MONITOR: timedelta(weeks=4),
            UrgencyLevel.ROUTINE: timedelta(days=365)  # 1 year
        }
        
        delta = timing_map.get(urgency, timedelta(days=365))
        recommended_date = now + delta
        
        return {
            "recommended_date": recommended_date.isoformat(),
            "urgency": urgency.value,
            "description": f"Schedule appointment by {recommended_date.strftime('%B %d, %Y')}"
        }
    
    def _get_educational_resources(self, condition: str) -> List[Dict]:
        """Get educational resources for the condition"""
        resources = {
            "diabetic_retinopathy": [
                {
                    "title": "American Diabetes Association - Eye Complications",
                    "url": "https://diabetes.org/diabetes/complications/eye-complications",
                    "description": "Comprehensive guide on diabetic eye disease"
                },
                {
                    "title": "National Eye Institute - Diabetic Retinopathy",
                    "url": "https://nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy",
                    "description": "Detailed information about diabetic retinopathy"
                }
            ],
            "glaucoma": [
                {
                    "title": "Glaucoma Foundation - Patient Education",
                    "url": "https://glaucomafoundation.org",
                    "description": "Educational materials about glaucoma management"
                }
            ],
            "cataract": [
                {
                    "title": "American Academy of Ophthalmology - Cataracts",
                    "url": "https://aao.org/eye-health/diseases/cataracts",
                    "description": "Patient guide to understanding cataracts"
                }
            ]
        }
        
        return resources.get(condition.lower(), [
            {
                "title": "National Eye Institute",
                "url": "https://nei.nih.gov",
                "description": "General eye health information"
            }
        ])
    
    def _get_support_contacts(self) -> Dict:
        """Get support contact information"""
        return {
            "emergency": {
                "number": "911 (US) or local emergency services",
                "description": "For medical emergencies"
            },
            "eye_emergency": {
                "description": "Contact your ophthalmologist's emergency line or go to nearest emergency room"
            },
            "telehealth": {
                "description": "Many providers offer telehealth consultations for urgent concerns"
            },
            "support_groups": {
                "description": "Patient support groups available through condition-specific organizations"
            }
        }
    
    def _get_fallback_recommendations(self, condition: str, confidence: float) -> Dict:
        """Fallback recommendations for unrecognized conditions"""
        return {
            "condition": condition,
            "confidence": confidence,
            "severity_level": "unknown",
            "urgency_level": "prompt",
            "timestamp": datetime.utcnow().isoformat(),
            
            "immediate_actions": [
                f"Condition '{condition}' detected with {confidence:.1f}% confidence",
                "Schedule comprehensive eye examination with ophthalmologist",
                "Bring this AI screening result to your appointment"
            ],
            "treatment_options": [
                "Professional evaluation required for treatment planning",
                "Additional testing may be recommended"
            ],
            "lifestyle_recommendations": [
                "Protect eyes from UV radiation",
                "Maintain overall eye health",
                "Follow up promptly with eye care professional"
            ],
            "follow_up_care": [
                "Comprehensive eye examination within 2 weeks",
                "Specialist referral if indicated"
            ],
            
            "confidence_notes": [
                f"AI detected condition with {confidence:.1f}% confidence",
                "Professional medical evaluation required for confirmation"
            ]
        }


# Convenience function for backward compatibility
def get_recommendations(predicted_class: str, confidence: float, severity_indicators: Dict = None) -> List[str]:
    """
    Backward-compatible function that returns simple recommendation list
    
    Args:
        predicted_class: The predicted eye condition
        confidence: Prediction confidence (0-100)
        severity_indicators: Additional indicators (optional)
    
    Returns:
        List of recommendation strings
    """
    system = TreatmentRecommendationSystem()
    comprehensive = system.get_comprehensive_recommendations(predicted_class, confidence, severity_indicators)
    
    # Combine all recommendations into simple list for backward compatibility
    recommendations = []
    recommendations.extend(comprehensive.get("immediate_actions", []))
    recommendations.extend(comprehensive.get("treatment_options", []))
    recommendations.extend(comprehensive.get("lifestyle_recommendations", [])[:3])  # Limit to top 3
    recommendations.extend(comprehensive.get("confidence_notes", []))
    
    return recommendations[:8]  # Limit total recommendations to maintain response size