import React, { useState, useEffect, useContext } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Button, Alert, TouchableOpacity, ScrollView, Image } from 'react-native';
import { useRouter } from 'expo-router';
import { AuthContext } from '../../src/context/AuthContext';
import { getFeed, swipeItem, getDailyFeed, getSocialMatches } from '../../src/services/api';
import SwipeCard from '../../src/components/SwipeCard';

export default function HomeScreen() {
  const { logout, userToken, isLoading: authLoading } = useContext(AuthContext);

  // Modes: 'home', 'training', 'match', 'social'
  const [viewMode, setViewMode] = useState('home');

  const [feed, setFeed] = useState([]);
  const [loading, setLoading] = useState(false);

  // Daily Match Specifics
  const [dailyItem, setDailyItem] = useState(null);
  const [matchLoading, setMatchLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0); // 0-3

  // Social
  const [socialMatches, setSocialMatches] = useState([]);

  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !userToken) {
      router.replace('/login');
    }
  }, [userToken, authLoading]);

  // 1. TRAINING FEED (25 Items)
  const startTraining = async () => {
    setLoading(true);
    try {
      const items = await getFeed('movie');
      if (items && items.length > 0) {
        setFeed(items);
        setViewMode('training');
      } else {
        Alert.alert("All Caught Up", "No more training items for now.");
      }
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'Could not fetch training feed.');
    } finally {
      setLoading(false);
    }
  };

  // 2. DAILY RECOMMENDATION (Single Item + Animation)
  const fetchDailyRecommendation = async () => {
    setMatchLoading(true);

    // Step 1: Analyzing
    setLoadingStep(1);

    // Step 2: Checking Ratings (after 2.5s)
    setTimeout(() => setLoadingStep(2), 2500);

    // Step 3: Finding Gem (after 5s)
    setTimeout(() => setLoadingStep(3), 5000);

    try {
      // Use api.js service
      const item = await getDailyFeed();

      if (item) {
        // Wait for at least 6 seconds total
        setTimeout(() => {
          setDailyItem(item);
          setMatchLoading(false);
          setLoadingStep(0);
          setViewMode('match');
        }, 6000);
      } else {
        setMatchLoading(false);
        Alert.alert("Notice", "No daily recommendation available.");
      }

    } catch (e) {
      setMatchLoading(false);
      console.error(e);
      Alert.alert("Error", "Could not get daily match. " + (e.response?.data?.detail || e.message));
    }
  };

  // 3. SOCIAL MATCHES
  const fetchSocial = async () => {
    setLoading(true);
    try {
      // Use api.js service
      const data = await getSocialMatches();

      if (data) {
        setSocialMatches(data);
        setViewMode('social');
      }
    } catch (e) {
      console.error("Social Match Error:", e);
      Alert.alert('Error', `Request failed: ${e.response?.status} - ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSwipe = async (direction, item) => {
    setFeed((prev) => prev.filter((i) => i.id !== item.id));
    let action = 'dislike';
    if (direction === 'right') action = 'like';
    if (direction === 'up') action = 'superlike';
    if (direction === 'down') action = 'watchlist';
    try { await swipeItem(item.id, action); } catch (e) { }

    if (feed.length <= 1) {
      Alert.alert("Training Complete!", "Great job! Profile updated.");
      setViewMode('home');
    }
  };

  if (authLoading) return <ActivityIndicator style={styles.center} size="large" />;

  // --- LOADING SCREEN (Daily) ---
  if (matchLoading) {
    return (
      <View style={[styles.container, styles.center, { backgroundColor: '#000' }]}>
        <ActivityIndicator size="large" color="#00ff00" />
        <Text style={styles.loadingTitle}>AI ANALYZING...</Text>

        <Text style={styles.loadingText}>
          {loadingStep === 1 && "🧠 Scanning your taste profile..."}
          {loadingStep === 2 && "⭐ Cross-referencing TMDB ratings..."}
          {loadingStep === 3 && "💎 Found the perfect hidden gem!"}
        </Text>
      </View>
    )
  }

  // --- HOME ---
  if (viewMode === 'home') {
    return (
      <View style={styles.container}>
        <View style={styles.center}>
          <Text style={styles.title}>TasteMatch</Text>
          <Text style={styles.subtitle}>AI-Powered Movie Concierge</Text>

          <View style={styles.spacer} />

          <TouchableOpacity style={styles.mainButton} onPress={startTraining}>
            <Text style={styles.buttonText}>🏋️ Start Training</Text>
            <Text style={styles.subButtonText}>Swipe 25 movies to build profile</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.mainButton, styles.goldButton]} onPress={fetchDailyRecommendation}>
            <Text style={styles.buttonText}>🏆 Get Daily Recommendation</Text>
            <Text style={[styles.subButtonText, { color: 'black' }]}>Your single best movie for today</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.mainButton, styles.secondaryButton]} onPress={fetchSocial}>
            <Text style={styles.buttonText}>👥 Find My Soulmate</Text>
            <Text style={styles.subButtonText}>Match with others</Text>
          </TouchableOpacity>

          {loading && <ActivityIndicator size="large" color="#0a7ea4" style={{ marginTop: 20 }} />}
        </View>
      </View>
    );
  }

  // --- TRAINING ---
  if (viewMode === 'training') {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button title="Quit" onPress={() => setViewMode('home')} color="red" />
          <Text style={styles.headerTitle}>Training Session</Text>
          <View style={{ width: 50 }} />
        </View>
        <View style={styles.cardContainer}>
          {feed.map((item, index) => {
            if (index > 2) return null;
            return (
              <SwipeCard
                key={item.id}
                item={item}
                onSwipeLeft={() => handleSwipe('left', item)}
                onSwipeRight={() => handleSwipe('right', item)}
                onSwipeUp={() => handleSwipe('up', item)}
                onSwipeDown={() => handleSwipe('down', item)}
              />
            );
          }).reverse()}
        </View>
      </View>
    );
  }

  // --- MATCH RESULT ---
  if (viewMode === 'match' && dailyItem) {
    return (
      <View style={[styles.container, { backgroundColor: '#111' }]}>
        <ScrollView contentContainerStyle={styles.matchContainer}>
          <Text style={styles.matchTitle}>IT'S A MATCH!</Text>
          <Image source={{ uri: dailyItem.image_url }} style={styles.poster} />
          <Text style={styles.movieTitle}>{dailyItem.title}</Text>
          <View style={styles.badges}>
            <View style={styles.badge}><Text style={styles.badgeText}>AI Score: {dailyItem.match_score?.toFixed(1) || '?'}</Text></View>
            <View style={[styles.badge, { backgroundColor: '#f5c518' }]}><Text style={[styles.badgeText, { color: 'black' }]}>TMDB: {dailyItem.vote_average}</Text></View>
          </View>
          <Text style={styles.overview}>{dailyItem.overview}</Text>

          <TouchableOpacity style={[styles.mainButton, { marginTop: 20 }]} onPress={() => setViewMode('home')}>
            <Text style={styles.buttonText}>Great! Back to Home</Text>
          </TouchableOpacity>
        </ScrollView>
      </View>
    )
  }

  // --- SOCIAL ---
  if (viewMode === 'social') {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button title="< Home" onPress={() => setViewMode('home')} />
          <Text style={styles.headerTitle}>Soulmates</Text>
          <View style={{ width: 50 }} />
        </View>
        <ScrollView contentContainerStyle={styles.list}>
          {socialMatches.length === 0 ? <Text style={styles.text}>No soulmates found yet. Keep training!</Text> : null}
          {socialMatches.map((user, idx) => (
            <View key={idx} style={styles.userCard}>
              <View>
                <Text style={styles.username}>{user.username}</Text>
                <Text style={{ color: '#666' }}>ID: {user.user_id}</Text>
              </View>
              <Text style={styles.score}>{(user.similarity * 100).toFixed(0)}% Match</Text>
            </View>
          ))}
        </ScrollView>
      </View>
    );
  }

  return null;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f0f0f0' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 32, fontWeight: 'bold', color: '#333', marginBottom: 5 },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 40 },
  spacer: { height: 20 },

  mainButton: {
    backgroundColor: '#0a7ea4',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
    marginBottom: 15,
    elevation: 3,
  },
  goldButton: { backgroundColor: '#FFD700' },
  secondaryButton: { backgroundColor: '#4a4a4a' },

  buttonText: { color: 'white', fontSize: 18, fontWeight: 'bold' },
  subButtonText: { color: 'rgba(255,255,255,0.8)', fontSize: 12, marginTop: 2 },

  cardContainer: { flex: 1 },
  header: { padding: 10, paddingTop: 50, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: '#fff' },
  headerTitle: { fontSize: 18, fontWeight: 'bold' },
  list: { padding: 20 },
  userCard: { backgroundColor: 'white', padding: 20, borderRadius: 10, marginBottom: 10, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', elevation: 2 },
  username: { fontSize: 18, fontWeight: 'bold' },
  score: { fontSize: 18, color: '#0a7ea4', fontWeight: 'bold' },
  text: { fontSize: 16, color: '#666', textAlign: 'center', marginTop: 20 },

  loadingTitle: { fontSize: 24, color: '#fff', fontWeight: 'bold', marginTop: 20, marginBottom: 10 },
  loadingText: { fontSize: 16, color: '#ccc', fontStyle: 'italic' },

  matchContainer: { padding: 20, alignItems: 'center', paddingTop: 60 },
  matchTitle: { fontSize: 36, fontWeight: '900', color: '#E91E63', marginBottom: 20, letterSpacing: 2 },
  poster: { width: 300, height: 450, borderRadius: 15, marginBottom: 20, borderWidth: 2, borderColor: '#333' },
  movieTitle: { fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginBottom: 10, color: 'white' },
  badges: { flexDirection: 'row', marginBottom: 20 },
  badge: { backgroundColor: '#0a7ea4', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 20, marginHorizontal: 5 },
  badgeText: { color: 'white', fontWeight: 'bold' },
  overview: { fontSize: 14, color: '#ccc', textAlign: 'center', lineHeight: 22, marginBottom: 30 }
});
