import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, RefreshControl, Alert, TextInput } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import { getFriends, getFriendRequests, acceptFriendRequest, rejectFriendRequest } from '../../src/services/api';

export default function FriendsScreen() {
    const router = useRouter();
    const [friends, setFriends] = useState([]);
    const [requests, setRequests] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [activeTab, setActiveTab] = useState('list'); // 'list' or 'requests'

    const fetchData = async () => {
        setLoading(true);
        try {
            const [friendsData, requestsData] = await Promise.all([
                getFriends(),
                getFriendRequests()
            ]);
            setFriends(friendsData);
            setRequests(requestsData);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useFocusEffect(useCallback(() => {
        fetchData();
    }, []));

    const onRefresh = () => {
        setRefreshing(true);
        fetchData();
    };

    const handleAccept = async (requestId: number) => {
        try {
            await acceptFriendRequest(requestId);
            fetchData(); // Reload
        } catch (e) {
            Alert.alert("Error", "Failed to accept request");
        }
    };

    const handleReject = async (requestId: number) => {
        try {
            await rejectFriendRequest(requestId);
            fetchData();
        } catch (e) {
            Alert.alert("Error", "Failed to reject request");
        }
    };

    const renderFriendItem = ({ item }: { item: any }) => (
        <TouchableOpacity
            style={styles.card}
            onPress={() => router.push({
                pathname: '/matches/chat',
                params: { id: item.user_id, username: item.username }
            })}
        >
            <View style={styles.avatar}>
                <Text style={styles.avatarText}>{item.username.charAt(0).toUpperCase()}</Text>
            </View>
            <View style={styles.info}>
                <Text style={styles.username}>{item.username}</Text>
                <Text style={styles.subtext}>Tap to chat</Text>
            </View>
            <Ionicons name="chatbubble-outline" size={24} color="#FF3366" />
        </TouchableOpacity>
    );

    const renderRequestItem = ({ item }: { item: any }) => (
        <View style={styles.card}>
            <View style={styles.avatar}>
                <Text style={styles.avatarText}>{item.sender_username.charAt(0).toUpperCase()}</Text>
            </View>
            <View style={styles.info}>
                <Text style={styles.username}>{item.sender_username}</Text>
                <Text style={styles.subtext}>Sent a friend request</Text>
            </View>
            <View style={styles.actions}>
                <TouchableOpacity style={styles.acceptBtn} onPress={() => handleAccept(item.id)}>
                    <Ionicons name="checkmark" size={20} color="#fff" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.rejectBtn} onPress={() => handleReject(item.id)}>
                    <Ionicons name="close" size={20} color="#666" />
                </TouchableOpacity>
            </View>
        </View>
    );

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>Friends</Text>
                <TouchableOpacity onPress={() => router.push('/matches/add_friend')} style={styles.addBtn}>
                    <Ionicons name="person-add" size={22} color="#FF3366" />
                </TouchableOpacity>
            </View>

            <View style={styles.tabs}>
                <TouchableOpacity
                    style={[styles.tab, activeTab === 'list' && styles.activeTab]}
                    onPress={() => setActiveTab('list')}
                >
                    <Text style={[styles.tabText, activeTab === 'list' && styles.activeTabText]}>My Friends</Text>
                </TouchableOpacity>
                <TouchableOpacity
                    style={[styles.tab, activeTab === 'requests' && styles.activeTab]}
                    onPress={() => setActiveTab('requests')}
                >
                    <Text style={[styles.tabText, activeTab === 'requests' && styles.activeTabText]}>
                        Requests {requests.length > 0 && `(${requests.length})`}
                    </Text>
                </TouchableOpacity>
            </View>

            {activeTab === 'list' ? (
                <FlatList
                    data={friends}
                    renderItem={renderFriendItem}
                    keyExtractor={(item) => item.user_id.toString()}
                    contentContainerStyle={styles.list}
                    refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
                    ListEmptyComponent={<Text style={styles.emptyText}>No friends yet. Add some!</Text>}
                />
            ) : (
                <FlatList
                    data={requests}
                    renderItem={renderRequestItem}
                    keyExtractor={(item) => item.id.toString()}
                    contentContainerStyle={styles.list}
                    refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
                    ListEmptyComponent={<Text style={styles.emptyText}>No pending requests.</Text>}
                />
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        paddingTop: 50,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
        marginBottom: 10,
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        color: '#333',
    },
    addBtn: {
        padding: 5,
    },
    tabs: {
        flexDirection: 'row',
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    tab: {
        flex: 1,
        paddingVertical: 15,
        alignItems: 'center',
    },
    activeTab: {
        borderBottomWidth: 2,
        borderBottomColor: '#FF3366',
    },
    tabText: {
        fontSize: 16,
        color: '#999',
        fontWeight: '600',
    },
    activeTabText: {
        color: '#FF3366',
    },
    list: {
        padding: 20,
    },
    card: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#f9f9f9',
        padding: 15,
        borderRadius: 12,
        marginBottom: 10,
    },
    avatar: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: '#eee',
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 15,
    },
    avatarText: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#666',
    },
    info: {
        flex: 1,
    },
    username: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    subtext: {
        fontSize: 14,
        color: '#888',
    },
    actions: {
        flexDirection: 'row',
    },
    acceptBtn: {
        backgroundColor: '#4CAF50',
        padding: 8,
        borderRadius: 20,
        marginRight: 10,
    },
    rejectBtn: {
        backgroundColor: '#eee',
        padding: 8,
        borderRadius: 20,
    },
    emptyText: {
        textAlign: 'center',
        marginTop: 50,
        color: '#999',
        fontSize: 16,
    }
});
