import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, Button, ActivityIndicator, TouchableOpacity, Alert } from 'react-native';
import { useFocusEffect } from 'expo-router';
import { getSocialMatches } from '../../src/services/api';
import SocialSwipeCard from '../../src/components/SocialSwipeCard';
import { Ionicons } from '@expo/vector-icons';

export default function SocialScreen() {
    const [matches, setMatches] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useFocusEffect(
        useCallback(() => {
            fetchMatches();
        }, [])
    );

    const fetchMatches = async () => {
        setLoading(true);
        try {
            const data = await getSocialMatches();
            setMatches(data || []);
        } catch (e) {
            // Alert.alert("Error", "Could not load matches");
        } finally {
            setLoading(false);
        }
    };

    const handleSwipe = (direction: any, user: any) => {
        // Optimistic UI update
        setMatches(prev => prev.filter(u => u.user_id !== user.user_id));

        // TODO: Call API to record 'like' or 'dislike' on User
        // For now MVP just dismisses
        console.log(`Swiped ${direction} on ${user.username}`);
    };

    if (loading) {
        return <View style={styles.center}><ActivityIndicator size="large" color="#FF3366" /></View>;
    }

    if (matches.length === 0) {
        return (
            <View style={styles.center}>
                <Ionicons name="people-outline" size={64} color="#ccc" />
                <Text style={styles.emptyText}>No matches found nearby.</Text>
                <Text style={styles.subText}>Try improving your taste profile by rating more movies!</Text>
                <TouchableOpacity style={styles.refreshBtn} onPress={fetchMatches}>
                    <Text style={styles.btnText}>Refresh</Text>
                </TouchableOpacity>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>Soulmates</Text>
            </View>

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
        borderBottomColor: '#eee'
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
