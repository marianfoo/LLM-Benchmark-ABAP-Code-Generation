CLASS ltcl_test_common DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_common IMPLEMENTATION.
  METHOD test_case_1.
    DATA: l1       TYPE int4_table,
          l2       TYPE int4_table,
          expected TYPE int4_table,
          result   TYPE int4_table.

    l1 = VALUE #( ( 1 ) ( 4 ) ( 3 ) ( 34 ) ( 653 ) ( 2 ) ( 5 ) ).
    l2 = VALUE #( ( 5 ) ( 7 ) ( 1 ) ( 5 ) ( 9 ) ( 653 ) ( 121 ) ).
    expected = VALUE #( ( 1 ) ( 5 ) ( 653 ) ).

    result = z_humaneval_058=>common( l1 = l1 l2 = l2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: l1       TYPE int4_table,
          l2       TYPE int4_table,
          expected TYPE int4_table,
          result   TYPE int4_table.

    l1 = VALUE #( ( 5 ) ( 3 ) ( 2 ) ( 8 ) ).
    l2 = VALUE #( ( 3 ) ( 2 ) ).
    expected = VALUE #( ( 2 ) ( 3 ) ).

    result = z_humaneval_058=>common( l1 = l1 l2 = l2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: l1       TYPE int4_table,
          l2       TYPE int4_table,
          expected TYPE int4_table,
          result   TYPE int4_table.

    l1 = VALUE #( ( 4 ) ( 3 ) ( 2 ) ( 8 ) ).
    l2 = VALUE #( ( 3 ) ( 2 ) ( 4 ) ).
    expected = VALUE #( ( 2 ) ( 3 ) ( 4 ) ).

    result = z_humaneval_058=>common( l1 = l1 l2 = l2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: l1       TYPE int4_table,
          l2       TYPE int4_table,
          expected TYPE int4_table,
          result   TYPE int4_table.

    l1 = VALUE #( ( 4 ) ( 3 ) ( 2 ) ( 8 ) ).
    l2 = VALUE #( ).
    expected = VALUE #( ).

    result = z_humaneval_058=>common( l1 = l1 l2 = l2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.
ENDCLASS.