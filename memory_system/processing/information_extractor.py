"""
Information extraction from text.

This module provides extraction of entities, facts, preferences,
and other structured information from text.
"""

import re
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict
from ..processing.base_processor import BaseProcessor
from ..core.models import MemoryFragment, MemoryType


class InformationExtractor(BaseProcessor):
    """
    Information extractor for structured data extraction.
    
    Extracts entities, facts, preferences, and other structured
    information from unstructured text.
    """
    
    def _initialize(self) -> None:
        """Initialize extractor components."""
        self.extract_entities = self.config.get('extract_entities', True)
        self.extract_facts = self.config.get('extract_facts', True)
        self.extract_preferences = self.config.get('extract_preferences', True)
        self.extract_skills = self.config.get('extract_skills', False)
        self.extract_dates = self.config.get('extract_dates', True)
        self.extract_numbers = self.config.get('extract_numbers', True)
        
        # Patterns for extraction
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for extraction."""
        # Entity patterns
        self.name_pattern = re.compile(r'\b([A-Z][a-z]+ (?:[A-Z][a-z]+ ?)+)\b')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        self.url_pattern = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)')
        
        # Fact patterns
        self.fact_patterns = [
            re.compile(r'([^.]+)\s+(?:is|are|was|were)\s+([^.]+)'),
            re.compile(r'([^.]+)\s+(?:has|have|had)\s+([^.]+)'),
            re.compile(r'([^.]+)\s+(?:can|could|will|would)\s+([^.]+)')
        ]
        
        # Preference patterns
        self.preference_patterns = [
            re.compile(r"I (?:like|love|enjoy|prefer)\s+([^.]+)", re.IGNORECASE),
            re.compile(r"I (?:don't|do not|dislike|hate)\s+([^.]+)", re.IGNORECASE),
            re.compile(r"My favorite\s+([^.]+)", re.IGNORECASE),
            re.compile(r"I'm (?:a|an)\s+([^.]+)", re.IGNORECASE),
            re.compile(r"I (?:want|need|would like)\s+([^.]+)", re.IGNORECASE)
        ]
        
        # Skill patterns
        self.skill_patterns = [
            re.compile(r"I (?:can|know how to)\s+([^.]+)", re.IGNORECASE),
            re.compile(r"I'm (?:good at|skilled in|experienced with)\s+([^.]+)", re.IGNORECASE),
            re.compile(r"I have experience (?:with|in)\s+([^.]+)", re.IGNORECASE)
        ]
        
        # Date patterns
        self.date_patterns = [
            re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'),
            re.compile(r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b'),
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})\b', re.IGNORECASE)
        ]
        
        # Number patterns
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.currency_pattern = re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?')
        self.percentage_pattern = re.compile(r'\b\d+(?:\.\d+)?%')
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Extract information from text.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with extracted information
        """
        results = {
            'entities': {},
            'facts': [],
            'preferences': [],
            'skills': [],
            'dates': [],
            'numbers': [],
            'metadata': {
                'text_length': len(text),
                'extraction_timestamp': datetime.now().isoformat()
            }
        }
        
        if self.extract_entities:
            results['entities'] = self.extract_entities_from_text(text)
        
        if self.extract_facts:
            results['facts'] = self.extract_facts_from_text(text)
        
        if self.extract_preferences:
            results['preferences'] = self.extract_preferences_from_text(text)
        
        if self.extract_skills:
            results['skills'] = self.extract_skills_from_text(text)
        
        if self.extract_dates:
            results['dates'] = self.extract_dates_from_text(text)
        
        if self.extract_numbers:
            results['numbers'] = self.extract_numbers_from_text(text)
        
        return results
    
    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Source text
            
        Returns:
            Dictionary of entity types and values
        """
        entities = {
            'names': [],
            'emails': [],
            'phones': [],
            'urls': [],
            'locations': [],
            'organizations': []
        }
        
        # Extract names
        names = self.name_pattern.findall(text)
        entities['names'] = list(set(names))
        
        # Extract emails
        emails = self.email_pattern.findall(text)
        entities['emails'] = list(set(emails))
        
        # Extract phone numbers
        phones = self.phone_pattern.findall(text)
        entities['phones'] = list(set(phones))
        
        # Extract URLs
        urls = self.url_pattern.findall(text)
        entities['urls'] = list(set(urls))
        
        # Extract locations (simple approach)
        location_keywords = ['Street', 'St', 'Avenue', 'Ave', 'Road', 'Rd', 'City', 'State']
        for keyword in location_keywords:
            pattern = re.compile(rf'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+{keyword})\b')
            locations = pattern.findall(text)
            entities['locations'].extend(locations)
        entities['locations'] = list(set(entities['locations']))
        
        # Extract organizations (simple approach)
        org_keywords = ['Inc', 'LLC', 'Corp', 'Corporation', 'Company', 'University', 'College']
        for keyword in org_keywords:
            pattern = re.compile(rf'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+{keyword})\b')
            orgs = pattern.findall(text)
            entities['organizations'].extend(orgs)
        entities['organizations'] = list(set(entities['organizations']))
        
        return entities
    
    def extract_facts_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract factual statements from text.
        
        Args:
            text: Source text
            
        Returns:
            List of extracted facts
        """
        facts = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern in self.fact_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    if len(match) == 2:
                        subject, predicate = match
                        fact = {
                            'subject': subject.strip(),
                            'predicate': predicate.strip(),
                            'sentence': sentence,
                            'confidence': 0.7  # Simple heuristic
                        }
                        facts.append(fact)
                        break
        
        return facts[:10]  # Limit to top 10 facts
    
    def extract_preferences_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract user preferences from text.
        
        Args:
            text: Source text
            
        Returns:
            List of extracted preferences
        """
        preferences = []
        
        for pattern in self.preference_patterns:
            matches = pattern.findall(text)
            for match in matches:
                preference = {
                    'content': match.strip(),
                    'type': self._classify_preference(match),
                    'sentiment': self._get_preference_sentiment(pattern.pattern),
                    'confidence': 0.8
                }
                preferences.append(preference)
        
        # Deduplicate similar preferences
        unique_preferences = []
        seen_content = set()
        
        for pref in preferences:
            content_lower = pref['content'].lower()
            if content_lower not in seen_content:
                unique_preferences.append(pref)
                seen_content.add(content_lower)
        
        return unique_preferences
    
    def extract_skills_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract skills from text.
        
        Args:
            text: Source text
            
        Returns:
            List of extracted skills
        """
        skills = []
        
        for pattern in self.skill_patterns:
            matches = pattern.findall(text)
            for match in matches:
                skill = {
                    'skill': match.strip(),
                    'proficiency': 'mentioned',
                    'confidence': 0.7
                }
                skills.append(skill)
        
        return skills
    
    def extract_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract dates from text.
        
        Args:
            text: Source text
            
        Returns:
            List of extracted dates
        """
        dates = []
        
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                date_dict = {
                    'raw': ' '.join(str(m) for m in match) if isinstance(match, tuple) else match,
                    'parsed': self._parse_date(match),
                    'context': self._get_date_context(text, match)
                }
                dates.append(date_dict)
        
        return dates
    
    def extract_numbers_from_text(self, text: str) -> Dict[str, List]:
        """
        Extract numbers from text.
        
        Args:
            text: Source text
            
        Returns:
            Dictionary of number types and values
        """
        numbers = {
            'regular': [],
            'currency': [],
            'percentage': []
        }
        
        # Extract currency
        currency_matches = self.currency_pattern.findall(text)
        numbers['currency'] = list(set(currency_matches))
        
        # Extract percentages
        percentage_matches = self.percentage_pattern.findall(text)
        numbers['percentage'] = list(set(percentage_matches))
        
        # Extract regular numbers (excluding those in currency/percentage)
        all_numbers = self.number_pattern.findall(text)
        regular_numbers = []
        for num in all_numbers:
            if not any(num in c for c in numbers['currency']) and \
               not any(num in p for p in numbers['percentage']):
                regular_numbers.append(num)
        numbers['regular'] = list(set(regular_numbers))[:20]  # Limit to 20
        
        return numbers
    
    def create_memory_fragments(self, extraction_results: Dict[str, Any],
                              user_id: str) -> List[MemoryFragment]:
        """
        Create memory fragments from extraction results.
        
        Args:
            extraction_results: Results from extraction
            user_id: User ID for the memories
            
        Returns:
            List of MemoryFragment objects
        """
        fragments = []
        
        # Create fragments for entities
        if extraction_results.get('entities'):
            for entity_type, values in extraction_results['entities'].items():
                for value in values:
                    if value:  # Skip empty values
                        fragment = MemoryFragment(
                            user_id=user_id,
                            fragment_type=MemoryType.ENTITY,
                            content=f"{entity_type}: {value}",
                            importance_score=0.7
                        )
                        fragments.append(fragment)
        
        # Create fragments for facts
        for fact in extraction_results.get('facts', []):
            fragment = MemoryFragment(
                user_id=user_id,
                fragment_type=MemoryType.FACT,
                content=f"{fact['subject']} {fact['predicate']}",
                importance_score=fact.get('confidence', 0.5)
            )
            fragments.append(fragment)
        
        # Create fragments for preferences
        for pref in extraction_results.get('preferences', []):
            fragment = MemoryFragment(
                user_id=user_id,
                fragment_type=MemoryType.PREFERENCE,
                content=pref['content'],
                importance_score=pref.get('confidence', 0.7),
                metadata={'type': pref['type'], 'sentiment': pref['sentiment']}
            )
            fragments.append(fragment)
        
        # Create fragments for skills
        for skill in extraction_results.get('skills', []):
            fragment = MemoryFragment(
                user_id=user_id,
                fragment_type=MemoryType.SKILL,
                content=skill['skill'],
                importance_score=skill.get('confidence', 0.6)
            )
            fragments.append(fragment)
        
        return fragments
    
    def _classify_preference(self, preference_text: str) -> str:
        """Classify preference type."""
        preference_lower = preference_text.lower()
        
        if any(word in preference_lower for word in ['food', 'eat', 'drink', 'taste']):
            return 'food'
        elif any(word in preference_lower for word in ['music', 'song', 'listen', 'band']):
            return 'music'
        elif any(word in preference_lower for word in ['movie', 'film', 'watch', 'show']):
            return 'entertainment'
        elif any(word in preference_lower for word in ['read', 'book', 'author', 'novel']):
            return 'reading'
        elif any(word in preference_lower for word in ['sport', 'play', 'game', 'exercise']):
            return 'sports'
        elif any(word in preference_lower for word in ['travel', 'visit', 'place', 'country']):
            return 'travel'
        else:
            return 'general'
    
    def _get_preference_sentiment(self, pattern_text: str) -> str:
        """Get sentiment of preference pattern."""
        if any(word in pattern_text for word in ["don't", "not", "dislike", "hate"]):
            return 'negative'
        elif any(word in pattern_text for word in ["love", "favorite"]):
            return 'strong_positive'
        else:
            return 'positive'
    
    def _parse_date(self, date_match) -> Optional[str]:
        """Parse date match into ISO format."""
        try:
            # This is a simplified parser
            # In production, use dateutil.parser
            if isinstance(date_match, tuple):
                # Convert tuple to string representation
                date_str = '-'.join(str(part) for part in date_match)
                return date_str
            return str(date_match)
        except Exception:
            return None
    
    def _get_date_context(self, text: str, date_match, window: int = 50) -> str:
        """Get context around a date mention."""
        date_str = ' '.join(str(m) for m in date_match) if isinstance(date_match, tuple) else str(date_match)
        
        # Find position of date in text
        pos = text.find(date_str)
        if pos == -1:
            return ""
        
        # Extract context window
        start = max(0, pos - window)
        end = min(len(text), pos + len(date_str) + window)
        
        context = text[start:end]
        
        # Clean up context
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context.strip()
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Source text
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Patterns for relationships
        relationship_patterns = [
            re.compile(r'(\w+)\s+(?:works for|employed by)\s+(\w+)'),
            re.compile(r'(\w+)\s+(?:married to|spouse of)\s+(\w+)'),
            re.compile(r'(\w+)\s+(?:manages|supervises|leads)\s+(\w+)'),
            re.compile(r'(\w+)\s+(?:owns|founded|created)\s+(\w+)')
        ]
        
        for pattern in relationship_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) == 2:
                    relationship = {
                        'subject': match[0],
                        'object': match[1],
                        'relation': pattern.pattern.split('(?:')[1].split(')')[0]
                    }
                    relationships.append(relationship)
        
        return relationships
