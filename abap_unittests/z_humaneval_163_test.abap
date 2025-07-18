CLASS ltc_test_generate_integers DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.

ENDCLASS.

CLASS ltc_test_generate_integers IMPLEMENTATION.

  METHOD test_case_1.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 2 ) ( 4 ) ( 6 ) ( 8 ) ).
    result = z_humaneval_163=>generate_integers( a = 2 b = 10 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 2 ) ( 4 ) ( 6 ) ( 8 ) ).
    result = z_humaneval_163=>generate_integers( a = 10 b = 2 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 2 ) ( 4 ) ( 6 ) ( 8 ) ).
    result = z_humaneval_163=>generate_integers( a = 132 b = 2 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test 3 failed'
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ).
    result = z_humaneval_163=>generate_integers( a = 17 b = 89 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test 4 failed'
    ).
  ENDMETHOD.

ENDCLASS.