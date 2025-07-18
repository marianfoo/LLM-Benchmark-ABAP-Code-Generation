CLASS ltcl_test_vowels_count DEFINITION FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_1 FOR TESTING,
      test_2 FOR TESTING,
      test_3 FOR TESTING,
      test_4 FOR TESTING,
      test_5 FOR TESTING,
      test_6 FOR TESTING,
      test_7 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_vowels_count IMPLEMENTATION.

  METHOD test_1.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'abcde' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 2
      msg = 'Test 1 failed'
    ).
  ENDMETHOD.

  METHOD test_2.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'Alone' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 3
      msg = 'Test 2 failed'
    ).
  ENDMETHOD.

  METHOD test_3.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'key' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 2
      msg = 'Test 3 failed'
    ).
  ENDMETHOD.

  METHOD test_4.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'bye' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 1
      msg = 'Test 4 failed'
    ).
  ENDMETHOD.

  METHOD test_5.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'keY' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 2
      msg = 'Test 5 failed'
    ).
  ENDMETHOD.

  METHOD test_6.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'bYe' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 1
      msg = 'Test 6 failed'
    ).
  ENDMETHOD.

  METHOD test_7.
    DATA: result TYPE i.
    result = z_humaneval_064=>vowels_count( 'ACEDY' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 3
      msg = 'Test 7 failed'
    ).
  ENDMETHOD.

ENDCLASS.