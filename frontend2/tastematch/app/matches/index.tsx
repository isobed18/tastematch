import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, Image, ActivityIndicator, Alert } from 'react-native';
import { useRouter, useFocusEffect } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { getConfirmedMatches } from '../../src/services/api';

export default function MatchesScreen() {
    const [matches, setMatches] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useFocusEffect(
        useCallback(() => {
            fetchMatches();
        }, [])
    );

    const fetchMatches = async () => {
        try {
            setLoading(true);
            const data = await getConfirmedMatches();
            setMatches(data || []);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const renderItem = ({ item }: { item: any }) => (
        <TouchableOpacity
            style={styles.matchItem}
            onPress={() => router.push({
                pathname: '/matches/chat',
                params: { id: item.user_id, username: item.username }
            })}
        >
            <View style={styles.avatarContainer}>
                <View style={styles.placeholderAvatar}>
                    <Text style={styles.initial}>{item.username[0].toUpperCase()}</Text>
                </View>
                <View style={styles.onlineDot} />
            </View>

            <View style={styles.matchInfo}>
                <Text style={styles.username}>{item.username}</Text>
                <Text style={styles.lastMsg} numberOfLines={1}>
                    {item.match_reason || "Start a conversation!"}
                </Text>
            </View>

            <View style={styles.meta}>
                <Text style={styles.matchPercent}>{(item.similarity * 100).toFixed(0)}%</Text>
                <Ionicons name="chevron-forward" size={20} color="#ccc" />
            </View>
        </TouchableOpacity>
    );

    if (loading) {
        return <View style={styles.center}><ActivityIndicator color="#FF3366" /></View>;
    }

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                    <Ionicons name="arrow-back" size={24} color="#333" />
                </TouchableOpacity>
                <Text style={styles.title}>Matches ({matches.length})</Text>
            </View>

            {matches.length === 0 ? (
                <View style={styles.emptyState}>
                    <Ionicons name="heart-dislike-outline" size={60} color="#ccc" />
                    <Text style={styles.emptyText}>No matches yet.</Text>
                    <Text style={styles.subText}>Keep swiping to find your soulmate!</Text>
                </View>
            ) : (
                <FlatList
                    data={matches}
                    renderItem={renderItem}
                    keyExtractor={item => item.user_id.toString()}
                    contentContainerStyle={styles.listContent}
                />
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    center: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center'
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingTop: 50,
        paddingBottom: 15,
        paddingHorizontal: 20,
        borderBottomWidth: 1,
        borderBottomColor: '#f0f0f0',
        backgroundColor: '#fff'
    },
    backBtn: {
        marginRight: 15
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#333'
    },
    listContent: {
        padding: 20
    },
    matchItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: '#f5f5f5'
    },
    avatarContainer: {
        position: 'relative',
        marginRight: 15
    },
    placeholderAvatar: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: '#ffebee',
        justifyContent: 'center',
        alignItems: 'center'
    },
    initial: {
        fontSize: 20,
        color: '#FF3366',
        fontWeight: 'bold'
    },
    onlineDot: {
        width: 12,
        height: 12,
        borderRadius: 6,
        backgroundColor: '#4CAF50',
        position: 'absolute',
        bottom: 0,
        right: 0,
        borderWidth: 2,
        borderColor: '#fff'
    },
    matchInfo: {
        flex: 1
    },
    username: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 4
    },
    lastMsg: {
        fontSize: 14,
        color: '#777'
    },
    meta: {
        alignItems: 'flex-end'
    },
    matchPercent: {
        fontSize: 12,
        fontWeight: 'bold',
        color: '#FF3366',
        marginBottom: 5
    },
    emptyState: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 40
    },
    emptyText: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#555',
        marginTop: 15
    },
    subText: {
        textAlign: 'center',
        color: '#999',
        marginTop: 10
    }
});
