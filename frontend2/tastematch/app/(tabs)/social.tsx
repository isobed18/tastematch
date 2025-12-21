import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, Button, ActivityIndicator, TouchableOpacity, Alert } from 'react-native';
import { useFocusEffect, useRouter } from 'expo-router';
import { getSocialMatches, swipeUser } from '../../src/services/api';
import SocialSwipeCard from '../../src/components/SocialSwipeCard';
import MatchModal from '../../src/components/MatchModal';
import { Ionicons } from '@expo/vector-icons';

export default function SocialScreen() {
    const [matches, setMatches] = useState<any[]>([]);
    const [loading, setLoading] = useState(false); // Default to false, wait for user action
    const [started, setStarted] = useState(false); // To track if user clicked "Find Match"

    // Match Modal State
    const [showMatchModal, setShowMatchModal] = useState(false);
    const [matchedUser, setMatchedUser] = useState<any>(null);

    const router = useRouter();

    useFocusEffect(
        useCallback(() => {
            // Optional: Auto-fetch or wait for button?
            // User asked for a button. Let's wait.
        }, [])
    );

    const fetchMatches = async () => {
        setLoading(true);
        setStarted(true);
        try {
            const data = await getSocialMatches();
            setMatches(data || []);
        } catch (e) {
            // Alert.alert("Error", "Could not load matches");
        } finally {
            setLoading(false);
        }
    };

    const handleSwipe = async (direction: any, user: any) => {
        // Optimistic UI update
        const action = direction === 'right' ? 'like' : 'pass';
        setMatches(prev => prev.filter(u => u.user_id !== user.user_id));

        try {
            console.log(`Swiping ${action} on ${user.username}...`);
            const result = await swipeUser(user.user_id, action);

            if (result.is_match) {
                // Trigger Match Modal
                setMatchedUser(user); // Pass user object for display (we can enrich it if needed)
                setShowMatchModal(true);
            }
        } catch (error) {
            console.error("Swipe failed:", error);
        }
    };

    if (loading) {
        return (
            <View style={styles.container}>
                <View style={styles.header}>
                    <Text style={styles.title}>Soulmates</Text>
                    <TouchableOpacity onPress={() => router.push('/matches')}>
                        <Ionicons name="chatbubbles-outline" size={28} color="#FF3366" />
                    </TouchableOpacity>
                </View>
                <View style={styles.center}><ActivityIndicator size="large" color="#FF3366" /></View>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>Soulmates</Text>
                <TouchableOpacity onPress={() => router.push('/matches')}>
                    <Ionicons name="chatbubbles-outline" size={28} color="#FF3366" />
                </TouchableOpacity>
            </View>

            {matches.length === 0 ? (
                <View style={styles.center}>
                    <Ionicons name="heart-circle-outline" size={80} color="#FF3366" />
                    <Text style={styles.emptyText}>
                        {started ? "No more people nearby." : "Ready to meet your soulmate?"}
                    </Text>

                    {!started ? (
                        <Text style={styles.subText}>Find people with compatible movie & game tastes.</Text>
                    ) : (
                        <Text style={styles.subText}>Check back later for new members.</Text>
                    )}

                    <TouchableOpacity style={styles.refreshBtn} onPress={fetchMatches}>
                        <Text style={styles.btnText}>
                            {started ? "Refresh" : "Find My Match"}
                        </Text>
                    </TouchableOpacity>
                </View>
            ) : (
                <View style={styles.deck}>
                    {matches.map((user, index) => {
                        // Only render top 3 cards for performance
                        if (index > 2) return null;
                        return (
                            <SocialSwipeCard
                                key={user.user_id}
                                item={user}
                                onSwipeLeft={() => handleSwipe('left', user)}
                                onSwipeRight={() => handleSwipe('right', user)}
                            />
                        );
                    }).reverse()}
                </View>
            )}

            {/* Match Modal */}
            <MatchModal
                visible={showMatchModal}
                matchedUser={matchedUser}
                onClose={() => setShowMatchModal(false)}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
    },
    center: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20
    },
    header: {
        paddingTop: 60,
        paddingHorizontal: 20,
        paddingBottom: 10,
        backgroundColor: 'white',
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center'
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#FF3366'
    },
    deck: {
        flex: 1,
    },
    emptyText: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#333',
        marginTop: 20
    },
    subText: {
        textAlign: 'center',
        color: '#666',
        marginTop: 10,
        marginBottom: 30
    },
    refreshBtn: {
        backgroundColor: '#FF3366',
        paddingHorizontal: 30,
        paddingVertical: 12,
        borderRadius: 25
    },
    btnText: {
        color: 'white',
        fontWeight: 'bold'
    }
});
