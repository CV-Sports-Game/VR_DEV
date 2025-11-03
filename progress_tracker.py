"""
Progress Tracking and Session Storage System
===========================================
This module handles tracking user progress, storing session data, and generating
performance reports for VR sports training.

Features:
- Session recording and storage
- Progress tracking over time
- Performance metrics calculation
- Historical data analysis
- Export to JSON/CSV formats
"""

import os
import json
import csv
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

class ProgressTracker:
    """Tracks user progress and session data for sports training."""
    
    def __init__(self, db_path='progress.db', sessions_dir='sessions'):
        self.db_path = db_path
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for progress tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds REAL,
                total_frames INTEGER,
                analyzed_frames INTEGER,
                sport_type TEXT,
                metadata TEXT
            )
        ''')
        
        # Frame analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frame_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                frame_number INTEGER,
                timestamp REAL,
                pose_label TEXT,
                confidence REAL,
                model_type TEXT,
                sequence_pose_label TEXT,
                sequence_confidence REAL,
                feedback TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        # Progress metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                date DATE,
                total_sessions INTEGER,
                total_duration_seconds REAL,
                average_confidence REAL,
                poses_practiced TEXT,
                improvements TEXT,
                FOREIGN KEY (user_id) REFERENCES sessions(user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, user_id='default', sport_type='mixed', metadata=None):
        """Start a new training session."""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.now().isoformat(),
            'sport_type': sport_type,
            'metadata': metadata or {}
        }
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, start_time, sport_type, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, user_id, session_data['start_time'], sport_type, 
              json.dumps(metadata or {})))
        conn.commit()
        conn.close()
        
        # Save to JSON file
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return session_id
    
    def end_session(self, session_id, total_frames=0, analyzed_frames=0):
        """End a training session and calculate metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get start time
        cursor.execute('SELECT start_time FROM sessions WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
        
        start_time = datetime.fromisoformat(result[0])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update session
        cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, duration_seconds = ?, total_frames = ?, analyzed_frames = ?
            WHERE session_id = ?
        ''', (end_time.isoformat(), duration, total_frames, analyzed_frames, session_id))
        
        # Calculate session metrics
        cursor.execute('''
            SELECT pose_label, confidence, sequence_pose_label, sequence_confidence
            FROM frame_analyses
            WHERE session_id = ?
        ''', (session_id,))
        
        analyses = cursor.fetchall()
        
        conn.commit()
        conn.close()
        
        # Calculate and return metrics
        metrics = self._calculate_session_metrics(analyses)
        
        # Update session JSON file
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            session_data['end_time'] = end_time.isoformat()
            session_data['duration_seconds'] = duration
            session_data['total_frames'] = total_frames
            session_data['analyzed_frames'] = analyzed_frames
            session_data['metrics'] = metrics
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        
        return metrics
    
    def add_frame_analysis(self, session_id, frame_number, timestamp, 
                          pose_label, confidence, model_type='cnn',
                          sequence_pose_label=None, sequence_confidence=None,
                          feedback=None):
        """Add a frame analysis result to the session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO frame_analyses 
            (session_id, frame_number, timestamp, pose_label, confidence, 
             model_type, sequence_pose_label, sequence_confidence, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, frame_number, timestamp, pose_label, confidence,
              model_type, sequence_pose_label, sequence_confidence, feedback))
        
        conn.commit()
        conn.close()
    
    def _calculate_session_metrics(self, analyses):
        """Calculate metrics from frame analyses."""
        if not analyses:
            return {}
        
        pose_counts = defaultdict(int)
        confidences = []
        sequence_confidences = []
        poses_practiced = set()
        
        for analysis in analyses:
            pose_label, confidence, seq_pose_label, seq_confidence = analysis
            
            if pose_label:
                pose_counts[pose_label] += 1
                poses_practiced.add(pose_label)
                confidences.append(confidence or 0)
            
            if seq_pose_label:
                poses_practiced.add(seq_pose_label)
                sequence_confidences.append(seq_confidence or 0)
        
        metrics = {
            'total_frames_analyzed': len(analyses),
            'unique_poses': len(poses_practiced),
            'poses_practiced': sorted(list(poses_practiced)),
            'pose_distribution': dict(pose_counts),
            'average_confidence': np.mean(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'average_sequence_confidence': np.mean(sequence_confidences) if sequence_confidences else 0,
            'most_practiced_pose': max(pose_counts.items(), key=lambda x: x[1])[0] if pose_counts else None
        }
        
        return metrics
    
    def get_user_progress(self, user_id='default', days=30):
        """Get progress summary for a user over specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get sessions
        cursor.execute('''
            SELECT session_id, start_time, duration_seconds, total_frames, analyzed_frames
            FROM sessions
            WHERE user_id = ? AND start_time >= ?
            ORDER BY start_time DESC
        ''', (user_id, cutoff_date))
        
        sessions = cursor.fetchall()
        
        # Get all frame analyses for these sessions
        session_ids = [s[0] for s in sessions]
        if not session_ids:
            conn.close()
            return {}
        
        placeholders = ','.join('?' * len(session_ids))
        cursor.execute(f'''
            SELECT pose_label, confidence, sequence_pose_label, sequence_confidence
            FROM frame_analyses
            WHERE session_id IN ({placeholders})
        ''', session_ids)
        
        all_analyses = cursor.fetchall()
        conn.close()
        
        # Calculate overall progress
        progress = {
            'user_id': user_id,
            'period_days': days,
            'total_sessions': len(sessions),
            'total_duration_hours': sum(s[3] or 0 for s in sessions) / 3600,
            'total_frames_analyzed': sum(s[4] or 0 for s in sessions),
            'sessions': []
        }
        
        # Per-session metrics
        for session in sessions:
            session_id, start_time, duration, total_frames, analyzed_frames = session
            
            # Get analyses for this session
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pose_label, confidence, sequence_pose_label, sequence_confidence
                FROM frame_analyses
                WHERE session_id = ?
            ''', (session_id,))
            session_analyses = cursor.fetchall()
            
            metrics = self._calculate_session_metrics(session_analyses)
            
            progress['sessions'].append({
                'session_id': session_id,
                'start_time': start_time,
                'duration_seconds': duration,
                'metrics': metrics
            })
        
        # Overall trends
        if len(sessions) > 1:
            # Calculate improvement trends
            recent_sessions = sessions[:min(5, len(sessions))]
            older_sessions = sessions[min(5, len(sessions)):]
            
            if older_sessions:
                recent_avg_conf = np.mean([
                    self._calculate_session_metrics([
                        a for a in all_analyses 
                        if any(a[0] == s[0] for s in recent_sessions)
                    ]).get('average_confidence', 0)
                ])
                
                older_avg_conf = np.mean([
                    self._calculate_session_metrics([
                        a for a in all_analyses 
                        if any(a[0] == s[0] for s in older_sessions)
                    ]).get('average_confidence', 0)
                ])
                
                progress['improvement_trend'] = {
                    'confidence_change': recent_avg_conf - older_avg_conf,
                    'improving': recent_avg_conf > older_avg_conf
                }
        
        return progress
    
    def export_session(self, session_id, format='json'):
        """Export a session to file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session data
        cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            conn.close()
            return None
        
        # Get all frame analyses
        cursor.execute('SELECT * FROM frame_analyses WHERE session_id = ?', (session_id,))
        analyses = cursor.fetchall()
        conn.close()
        
        # Format data
        if format == 'json':
            session_data = {
                'session_id': session_row[0],
                'user_id': session_row[1],
                'start_time': session_row[2],
                'end_time': session_row[3],
                'duration_seconds': session_row[4],
                'total_frames': session_row[5],
                'analyzed_frames': session_row[6],
                'sport_type': session_row[7],
                'metadata': json.loads(session_row[8]) if session_row[8] else {},
                'frame_analyses': [
                    {
                        'frame_number': a[2],
                        'timestamp': a[3],
                        'pose_label': a[4],
                        'confidence': a[5],
                        'model_type': a[6],
                        'sequence_pose_label': a[7],
                        'sequence_confidence': a[8],
                        'feedback': a[9]
                    }
                    for a in analyses
                ]
            }
            
            output_file = os.path.join(self.sessions_dir, f"{session_id}_export.json")
            with open(output_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return output_file
        
        elif format == 'csv':
            output_file = os.path.join(self.sessions_dir, f"{session_id}_export.csv")
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_number', 'timestamp', 'pose_label', 'confidence',
                    'model_type', 'sequence_pose_label', 'sequence_confidence', 'feedback'
                ])
                
                for a in analyses:
                    writer.writerow([
                        a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]
                    ])
            
            return output_file
        
        return None
    
    def get_statistics(self, user_id='default'):
        """Get overall statistics for a user."""
        progress = self.get_user_progress(user_id, days=365)
        
        stats = {
            'user_id': user_id,
            'total_sessions': progress.get('total_sessions', 0),
            'total_training_hours': progress.get('total_duration_hours', 0),
            'total_frames_analyzed': progress.get('total_frames_analyzed', 0),
            'all_time_poses': set(),
            'recent_sessions': len([s for s in progress.get('sessions', []) 
                                  if datetime.fromisoformat(s['start_time']) > 
                                  datetime.now() - timedelta(days=7)])
        }
        
        # Collect all poses from all sessions
        for session in progress.get('sessions', []):
            metrics = session.get('metrics', {})
            stats['all_time_poses'].update(metrics.get('poses_practiced', []))
        
        stats['all_time_poses'] = sorted(list(stats['all_time_poses']))
        
        return stats


def main():
    """Example usage of ProgressTracker."""
    tracker = ProgressTracker()
    
    # Start a session
    session_id = tracker.start_session(user_id='user1', sport_type='boxing')
    print(f"Started session: {session_id}")
    
    # Simulate adding frame analyses
    for i in range(10):
        tracker.add_frame_analysis(
            session_id=session_id,
            frame_number=i,
            timestamp=i * 0.033,  # ~30 FPS
            pose_label='boxing_punch',
            confidence=0.85 + np.random.random() * 0.1,
            model_type='cnn',
            sequence_pose_label='jab' if i > 5 else 'guard',
            sequence_confidence=0.80 + np.random.random() * 0.15
        )
    
    # End session
    metrics = tracker.end_session(session_id, total_frames=100, analyzed_frames=10)
    print(f"Session metrics: {json.dumps(metrics, indent=2)}")
    
    # Get user progress
    progress = tracker.get_user_progress('user1', days=30)
    print(f"\nUser progress: {json.dumps(progress, indent=2, default=str)}")
    
    # Export session
    export_file = tracker.export_session(session_id, format='json')
    print(f"\nExported session to: {export_file}")


if __name__ == "__main__":
    main()

