CLASS ltcl_test_hex_key DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_ab FOR TESTING,
      test_1077e FOR TESTING,
      test_abed1a33 FOR TESTING,
      test_2020 FOR TESTING,
      test_123456789abcdef0 FOR TESTING,
      test_long_string FOR TESTING,
      test_empty_string FOR TESTING.

ENDCLASS.

CLASS ltcl_test_hex_key IMPLEMENTATION.

  METHOD test_ab.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( 'AB' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 1
      msg = 'First test error' ).
  ENDMETHOD.

  METHOD test_1077e.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( '1077E' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 2
      msg = 'Second test error' ).
  ENDMETHOD.

  METHOD test_abed1a33.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( 'ABED1A33' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 4
      msg = 'Third test error' ).
  ENDMETHOD.

  METHOD test_2020.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( '2020' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 2
      msg = 'Fourth test error' ).
  ENDMETHOD.

  METHOD test_123456789abcdef0.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( '123456789ABCDEF0' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 6
      msg = 'Fifth test error' ).
  ENDMETHOD.

  METHOD test_long_string.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( '112233445566778899AABBCCDDEEFF00' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 12
      msg = 'Sixth test error' ).
  ENDMETHOD.

  METHOD test_empty_string.
    DATA: result TYPE i.
    result = z_humaneval_078=>hex_key( '' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 0
      msg = 'Empty string test error' ).
  ENDMETHOD.

ENDCLASS.