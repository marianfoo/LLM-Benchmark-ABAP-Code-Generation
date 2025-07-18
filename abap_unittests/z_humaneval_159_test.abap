CLASS ltc_test_eat DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING.

ENDCLASS.

CLASS ltc_test_eat IMPLEMENTATION.

  METHOD test_case_1.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    result = z_humaneval_159=>eat( number = 5 need = 6 remaining = 10 ).
    expected = VALUE #( ( 11 ) ( 4 ) ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    result = z_humaneval_159=>eat( number = 4 need = 8 remaining = 9 ).
    expected = VALUE #( ( 12 ) ( 1 ) ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    result = z_humaneval_159=>eat( number = 1 need = 10 remaining = 10 ).
    expected = VALUE #( ( 11 ) ( 0 ) ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    result = z_humaneval_159=>eat( number = 2 need = 11 remaining = 5 ).
    expected = VALUE #( ( 7 ) ( 0 ) ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    result = z_humaneval_159=>eat( number = 4 need = 5 remaining = 7 ).
    expected = VALUE #( ( 9 ) ( 2 ) ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    result = z_humaneval_159=>eat( number = 4 need = 5 remaining = 1 ).
    expected = VALUE #( ( 5 ) ( 0 ) ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

ENDCLASS.