CLASS ltcl_test_make_a_pile DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_make_a_pile_case_3 FOR TESTING,
      test_make_a_pile_case_4 FOR TESTING,
      test_make_a_pile_case_5 FOR TESTING,
      test_make_a_pile_case_6 FOR TESTING,
      test_make_a_pile_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_make_a_pile IMPLEMENTATION.

  METHOD test_make_a_pile_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_100=>make_a_pile( 3 )
      exp = VALUE int4_table( ( 3 ) ( 5 ) ( 7 ) )
      msg = 'Test 3' ).
  ENDMETHOD.

  METHOD test_make_a_pile_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_100=>make_a_pile( 4 )
      exp = VALUE int4_table( ( 4 ) ( 6 ) ( 8 ) ( 10 ) )
      msg = 'Test 4' ).
  ENDMETHOD.

  METHOD test_make_a_pile_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_100=>make_a_pile( 5 )
      exp = VALUE int4_table( ( 5 ) ( 7 ) ( 9 ) ( 11 ) ( 13 ) )
      msg = 'Test 5' ).
  ENDMETHOD.

  METHOD test_make_a_pile_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_100=>make_a_pile( 6 )
      exp = VALUE int4_table( ( 6 ) ( 8 ) ( 10 ) ( 12 ) ( 14 ) ( 16 ) )
      msg = 'Test 6' ).
  ENDMETHOD.

  METHOD test_make_a_pile_case_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_100=>make_a_pile( 8 )
      exp = VALUE int4_table( ( 8 ) ( 10 ) ( 12 ) ( 14 ) ( 16 ) ( 18 ) ( 20 ) ( 22 ) )
      msg = 'Test 8' ).
  ENDMETHOD.

ENDCLASS.