CLASS ltcl_test_will_it_fly DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_balanced_and_light FOR TESTING,
      test_unbalanced_and_light FOR TESTING,
      test_single_element_light FOR TESTING,
      test_balanced_and_heavy FOR TESTING,
      test_unbalanced_sequence FOR TESTING,
      test_single_element_exact FOR TESTING.

ENDCLASS.

CLASS ltcl_test_will_it_fly IMPLEMENTATION.

  METHOD test_balanced_and_light.
    DATA: q TYPE int4_table.
    q = VALUE #( ( 3 ) ( 2 ) ( 3 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_072=>will_it_fly( q = q w = 9 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_unbalanced_and_light.
    DATA: q TYPE int4_table.
    q = VALUE #( ( 1 ) ( 2 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_072=>will_it_fly( q = q w = 5 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_single_element_light.
    DATA: q TYPE int4_table.
    q = VALUE #( ( 3 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_072=>will_it_fly( q = q w = 5 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_balanced_and_heavy.
    DATA: q TYPE int4_table.
    q = VALUE #( ( 3 ) ( 2 ) ( 3 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_072=>will_it_fly( q = q w = 1 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_unbalanced_sequence.
    DATA: q TYPE int4_table.
    q = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_072=>will_it_fly( q = q w = 6 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_single_element_exact.
    DATA: q TYPE int4_table.
    q = VALUE #( ( 5 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_072=>will_it_fly( q = q w = 5 )
      exp = abap_true ).
  ENDMETHOD.

ENDCLASS.